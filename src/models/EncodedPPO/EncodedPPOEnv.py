from gymnasium import Env, spaces
import torch
import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from models.EncodedPPO.utils import mask_fn
from env.hearthstone.HearthVsAgentEnv import HearthVsAgentEnv

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

class EncodedPPOEnv(Env):
    """
    Simulated Hearthstone Environment using a world model.
    
    :param real_env: The real environment to interact with.
    :param device: The device to use for computation (e.g., "cpu" or "cuda").
    :param cont_indices: Indices of continuous features in the observation space.
    :param disc_indices: Indices of discrete features in the observation space.
    :param observation_shape: Shape of the observation space.
    """
    def __init__(
        self, 
        autoencoder, real_env: HearthVsAgentEnv, 
        device="cpu", cont_indices=None, disc_indices=None, 
        observation_shape=None
        ):
        super(EncodedPPOEnv, self).__init__()
        self.autoencoder = autoencoder
        self.real_env = real_env
        self.action_space = real_env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32
        )
        self.device = device
        self.cont_indices = cont_indices
        self.disc_indices = disc_indices

        self.reset()

    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation.
        
        :param kwargs: Additional arguments for the real environment's reset method.
        :return: Initial observation and info dictionary.
        """
        real_obs, _ = self.real_env.reset(**kwargs)
        with torch.no_grad():
            real_obs_tensor = torch.tensor(real_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            latent_state = self.autoencoder.encode(
                real_obs_tensor, 
                cont_idx=self.cont_indices, 
                disc_idx=self.disc_indices
                )
            
            observation = latent_state
        obs_np = to_numpy(observation.squeeze(0))
        info = {}
        return obs_np, info

    def step(self, action):
        """
        Take a step in the environment using the given action.
        
        :param action: The action to take.
        :return: Tuple of (observation, reward, done, truncated, info).
        """
        real_obs, reward, done, truncated, info = self.real_env.step(action)
        with torch.no_grad():
            real_obs_tensor = torch.tensor(real_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            latent_state = self.autoencoder.encode(
                real_obs_tensor, 
                cont_idx=self.cont_indices, 
                disc_idx=self.disc_indices
                )
            observation = latent_state
        obs_np = to_numpy(observation.squeeze(0))
        return obs_np, reward, done, truncated, info

    def render(self, mode='human'):
        return self.real_env.render(mode)
    
    def close(self):
        return self.real_env.close()
    
    def get_valid_actions(self):
        """
        Get valid actions and action mask from the real environment.
        
        :return: Tuple of valid actions and action mask.
        """
        # Use the real environment's action mask to get valid actions.
        valid_actions, action_mask = self.real_env.get_valid_actions()
        return valid_actions, action_mask
    
    def _update_opponent(self, opponent_agent):
        """
        Update the opponent agent.
        
        :param opponent_agent: The new opponent agent.
        """
        self.opponent_agent = opponent_agent
        
def make_env(
    autoencoder,
    real_env: HearthVsAgentEnv,
    device,
    cont_indices=None,
    disc_indices=None,
    observation_shape=None,
    ):
    """
    Create a simulated Hearthstone environment using a world model and a real environment.
    
    :param real_env: The real environment to interact with.
    :param device: The device to use for computation (e.g., "cpu" or "cuda").
    :param cont_indices: Indices of continuous features in the observation space.
    :param disc_indices: Indices of discrete features in the observation space.
    :param observation_shape: Shape of the observation space.
    :return: A wrapped environment with action masking.
    """
    # 1) Create your base environment
    env = EncodedPPOEnv(
        autoencoder, real_env=real_env, 
        device=device, cont_indices=cont_indices, 
        disc_indices=disc_indices, observation_shape=observation_shape
        )
    
    # 2) Wrap with ActionMasker if you need action masking
    env = ActionMasker(env, mask_fn)
    
    return env

