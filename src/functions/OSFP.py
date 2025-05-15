from stable_baselines3.common.callbacks import BaseCallback
from agents.HearthstoneAgent import HearthstoneAgent
from env.hearthstone.HearthGym import HearthstoneEnv
import numpy as np


def update_env_opponent(vec_env, opponent_agent):
    """Update the opponent agent in the environment."""
    for env in vec_env.envs:
        base_env = env.env
        base_env._update_opponent(opponent_agent)

class OSFPCallback(BaseCallback):
    """
    Callback for updating the average policy using Online Stochastic Fixed Point (OSFP) method.
    
    :param holder: An instance of AveragePolicyHolder that holds the average policy.
    :param alpha: The learning rate for the moving average.
    :param update_freq: Frequency of updating the average policy.
    :param verbose: Verbosity level.
    """
    def __init__(self, holder, alpha=0.01, update_freq=5000, verbose=0):
        super(OSFPCallback, self).__init__(verbose)
        self.holder = holder  # This is an AveragePolicyHolder instance
        self.alpha = alpha
        self.update_freq = update_freq
        self.last_update = 0

    def _on_step(self) -> bool:
        """
        Called at each step of the training process.
        :return: True to continue training, False to stop.
        """
        if self.num_timesteps - self.last_update >= self.update_freq:
            self._update_average_policy()
            self.last_update = self.num_timesteps
        return True

    def _update_average_policy(self):
        """	
        Update the average policy using a moving average of the current model parameters.
        """
        all_parameters = zip(self.model.policy.parameters(), self.holder.model.policy.parameters())
        for current_param, avg_param in all_parameters:
            avg_param.data.copy_((1 - self.alpha) * avg_param.data + self.alpha * current_param.data)
        if self.verbose > 0:
            print(f"OSFP: Updated average policy at timestep {self.num_timesteps}.")

            
class AveragePolicyHolder:
    """
    A class to hold the average policy model using the OSFP method.
    
    :param current_model: The current model to be averaged.
    :param model_type: The type of model to create for the average policy.
    :param model_policy: The policy to be used in the average model.
    :param policy_kwargs: Additional keyword arguments for the policy.
    :param env: The environment to be used for the average policy.
    :param device: The device to run the model on (e.g., "cpu" or "cuda").
    :param gamma: Discount factor for the model.
    :param gae_lambda: Lambda parameter for Generalized Advantage Estimation.
    :param n_steps: Number of steps to run for each environment per update.
    :param batch_size: Batch size for training.
    :param learning_rate: Learning rate for the optimizer.
    :param n_epochs: Number of epochs to train the model.
    :param seed: Random seed for reproducibility.
    :param clip_range: Clipping range for the PPO algorithm.
    :param ent_coef: Coefficient for the entropy term in the loss function.
    """
    def __init__(
        self, 
        current_model, model_type, model_policy, 
        policy_kwargs, env, device, 
        gamma, gae_lambda, n_steps, 
        batch_size, learning_rate, n_epochs, 
        seed, clip_range, ent_coef):
        
        # Create a new instance of the same model that will act as the average policy
        self.model = model_type(
            policy=model_policy,
            policy_kwargs=policy_kwargs,
            env=env,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            seed=seed,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=0,  # no verbosity for the holder
        )
        # Initialize the average policy using the current model's parameters
        self.model.policy.load_state_dict(current_model.policy.state_dict())


class AveragePolicyAgent(HearthstoneAgent):
    """
    An agent that uses an average policy for action selection.
    :param holder: An instance of AveragePolicyHolder that holds the average policy.
    """
    def __init__(self, holder):
        super().__init__()
        self.holder = holder  # This holder contains the average model

    def act(self, observation, valid_actions: list = None, action_mask = None, env: HearthstoneEnv = None) -> dict | np.ndarray:
        """
        Select an action based on the average policy.
        :param observation: The current observation from the environment.
        :param valid_actions: List of valid actions (not used in this implementation).
        :param action_mask: Mask for valid actions (not used in this implementation).
        :param env: The environment (not used in this implementation).
        :return: The selected action.
        """
        
        # Retrieve the action from the average policy model
        action, _ = self.holder.model.predict(observation, deterministic=True, action_masks=action_mask)
        return action


