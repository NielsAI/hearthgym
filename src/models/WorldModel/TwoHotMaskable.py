# models/PPO/twohot_maskable.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.type_aliases import Schedule
from models.WorldModel.utils import two_hot
import numpy as np

class TwoHotMaskablePolicy(MaskableActorCriticPolicy):
    """
    Maskable policy with two-hot distributional value head.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule: Schedule,
        return_bins: int = 255,
        **kwargs
    ):
        self.return_bins = return_bins
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        # build shared and actor net as usual
        super()._build_mlp_extractor()
        # override critic head to produce logits
        self.value_net = nn.Linear(self.features_dim, self.return_bins)
        self.mlp_extractor.value_net = self.value_net

    def _predict_values(self, obs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # output distribution logits
        return self.value_net(features)

class TwoHotMaskablePPO(MaskablePPO):
    """
    MaskablePPO that trains value head with two-hot cross-entropy.
    """
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: Schedule = 3e-4,
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Schedule = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 0,
        device: str = "auto",
        return_bins: int = 51,
        policy_kwargs: dict = None,
        **kwargs
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            device=device,
            policy_kwargs=policy_kwargs,
        )
        self.return_bins = return_bins
        # ensure schedule for clip_range
        self.clip_range = get_schedule_fn(self.clip_range)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer,
        with a two-hot distributional critic trained via KL divergence.
        """
        # 1) Switch to train mode, update LR schedules
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True

        # 2) Epoch loop
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            # 3) Batch loop
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # ——— Actions & values ——————————————————————————————————
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                # Evaluate policy: values now are logits [B, return_bins]
                value_logits, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                # ——— Policy loss ————————————————————————————————————
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                pg1   = advantages * ratio
                pg2   = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(pg1, pg2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(((ratio - 1).abs() > clip_range).float().mean().item())

                # ——— Distributional value loss ——————————————————————
                # 1) Possibly clip value change (not needed for distributional, but kept for compatibility)
                if self.clip_range_vf is not None:
                    # old_values here are logits as well—skip clipping for simplicity
                    value_logits_pred = value_logits
                else:
                    value_logits_pred = value_logits

                # 2) Build two-hot targets from the batch’s returns
                # rollout_data.returns is a tensor of shape (B,)
                twohot = two_hot(
                    rollout_data.returns.to(self.device),
                    v_min=-1.0, v_max=1.0,
                    num_bins=self.return_bins
                )  # shape [B, return_bins]

                # 3) KL divergence between model log-probs and two-hot target
                log_probs = F.log_softmax(value_logits_pred, dim=-1)  # [B, return_bins]
                value_loss = F.kl_div(log_probs, twohot, reduction='batchmean')
                value_losses.append(value_loss.item())

                # ——— Entropy loss ————————————————————————————————————
                if entropy is None:
                    entropy_loss = -log_prob.mean()
                else:
                    entropy_loss = -entropy.mean()
                entropy_losses.append(entropy_loss.item())

                # ——— Total loss & optimize ——————————————————————————
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Early stopping via approximate KL
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().cpu().item()
                    approx_kl_divs.append(approx_kl)
                    if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                        continue_training = False
                if not continue_training:
                    break

                # Gradient step
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        # 4) Logging
        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss",     np.mean(entropy_losses))
        self.logger.record("train/policy_gradient",  np.mean(pg_losses))
        self.logger.record("train/value_loss",       np.mean(value_losses))
        self.logger.record("train/approx_kl",        np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction",    np.mean(clip_fractions))
        self.logger.record("train/loss",             loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates",        self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range",       clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)