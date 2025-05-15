# dreamerv3_maskable_callback.py
from copy import deepcopy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class DreamerV3MaskableCallback(BaseCallback):
    """
    Composite callback that adds DreamerV3-style critic EMA and percentile return
    normalization.
    """
    def __init__(
        self,
        checkpoint_callback,
        osfp_callback=None,
        sigma: float = 0.98,
        return_percentile: float = 95.0,
        verbose: int = 0
    ):
        """Initialize the callback.
        :param checkpoint_callback: Checkpoint callback to be used.
        :param osfp_callback: Optional callback for Optimistic Smooth Fictitious Self-Play.
        :param sigma: Exponential moving average coefficient for the critic.
        :param return_percentile: Percentile for return normalization.
        :param verbose: Verbosity level.
        """
        # Initialize the base callback
        super().__init__(verbose)
        
        # Store the parameters
        self.checkpoint_cb = checkpoint_callback
        self.osfp_cb = osfp_callback
        self.sigma = sigma
        self.pct = return_percentile
        self.shadow_critic = None

    def _on_training_start(self):
        """Initialize the callback at the start of training."""
        # Initialize underlying callbacks
        self.checkpoint_cb.init_callback(self.model)
        if self.osfp_cb:
            self.osfp_cb.init_callback(self.model)
        # Clone critic to form EMA shadow network
        self.shadow_critic = deepcopy(self.model.policy)
        for param in self.shadow_critic.parameters():
            param.requires_grad_(False)

    def _on_rollout_end(self):
        """Update the shadow critic and normalize returns at the end of each rollout."""
        # Update EMA of critic weights
        live_params = list(self.model.policy.parameters())
        shadow_params = list(self.shadow_critic.parameters())
        for s_param, l_param in zip(shadow_params, live_params):
            s_param.data.mul_(self.sigma).add_(l_param.data * (1 - self.sigma))

        rb = self.model.rollout_buffer

        # Recompute advantages (normalized)
        values = rb.values
        advantages = rb.returns - values
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        rb.advantages = (advantages - adv_mean) / adv_std

        # 4. Proxy to underlying callbacks
        self.checkpoint_cb.on_rollout_end()
        if self.osfp_cb:
            self.osfp_cb.on_rollout_end()

    def _on_step(self) -> bool:
        """Perform a step in the training process."""
        # Proxy to underlying callbacks
        self.checkpoint_cb.on_step()
        if self.osfp_cb:
            self.osfp_cb.on_step()
        return True

    def _on_training_end(self):
        """Finalize the callback at the end of training."""
        # Proxy to underlying callbacks
        self.checkpoint_cb.on_training_end()
        if self.osfp_cb:
            self.osfp_cb.on_training_end()
