from gymnasium import Env, spaces
import numpy as np, torch
from sb3_contrib.common.wrappers import ActionMasker

from models.WorldModel.utils import one_hot_encode, symlog, symexp
from env.hearthstone.HearthVsAgentEnv import HearthVsAgentEnv
# ---------------------------------------------------------------------

def mask_fn(env):
    _, action_mask = env.get_valid_actions()

    # Double-check dtype
    action_mask = action_mask.astype(np.int8)

    return action_mask

def _vec(z_star: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Build controller feature vector.
    :param z_star: latent state (z*)
    :param h: hidden state (h)
    :return: concatenated vector of [symexp(z*), h] (Z+H,)
    """
    z = symexp(z_star.squeeze(0))
    return torch.cat([z, h.squeeze(0)], 0)


class DreamHearthstoneEnv(Env):
    """
    Dreamer-V3-inspired environment for Hearthstone.
    This environment wraps a real Hearthstone environment and uses a world model to simulate future states.
    Valid actions are determined by a Legality-Net, which is a neural network that predicts the legality of actions based on the current state.
    """

    def __init__(
        self,
        world_model,                
        real_env: HearthVsAgentEnv,
        device: str = "cpu",
        max_steps: int = 60,
        cont_indices=None,
        disc_indices=None,
        observation_shape=None,
        return_bins=255,  
        r_min=-1,
        r_max=1,            
    ):
        """	
        Initialize the DreamHearthstoneEnv.
        :param world_model: The world model used for simulating future states. (contains AutoEncoder, RSSM, and Legality-Net)
        :param real_env: The real Hearthstone environment used for initial observation.
        :param device: The device to use for computations (CPU or GPU).
        :param max_steps: The maximum number of steps in an episode.
        :param cont_indices: Indices of continuous features in the observation space.
        :param disc_indices: Indices of discrete features in the observation space.
        :param observation_shape: Shape of the observation space.
        :param return_bins: Number of bins for the reward distribution.
        :param r_min: Minimum value for the reward distribution.
        :param r_max: Maximum value for the reward distribution.
        """
        # Call parent constructor
        super().__init__()
        self.wm      = world_model
        self.realenv = real_env        # reset once; never stepped afterwards
        self.dev     = device
        self.max_T   = max_steps
        self.return_bins = return_bins
        self.r_min   = r_min
        self.r_max   = r_max

        # Set continuous and discrete indices
        self.cont_indices = cont_indices
        self.disc_indices = disc_indices

        self.action_space      = real_env.action_space
        self.observation_space = spaces.Box(
            np.inf, np.inf,
            observation_shape,
            dtype=np.float32
            )

        # internal counters
        self.turn = 0

    # -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state.
        :param seed: Random seed for reproducibility.
        :param options: Additional options for resetting the environment.
        :return: Initial observation and additional information.
        """
        # Reset the real environment and get the initial observation
        real_obs, _ = self.realenv.reset()
        real_t = torch.as_tensor(
            real_obs, dtype=torch.float32,
            device=self.dev).unsqueeze(0)

        # Encode the initial observation using the world model's autoencoder
        with torch.no_grad():
            z0 = self.wm.autoencoder.encode(real_t, self.cont_indices, self.disc_indices)
            init_state = self.wm.rssm.init_state(1, self.dev)
            self.state = (init_state[0], symlog(z0))

        # Set initial action, turn, done flag, and legality mask
        self.turn = 0

        feat    = _vec(self.state[1], self.state[0])  # (Z+H,)
        _, mask = self.realenv.get_valid_actions()              # real mask (1st)
        info    = {"action_mask": mask.astype(bool)}

        return feat.cpu().numpy().astype(np.float32), info

    def step(self, action_idx: int):
        """
        Perform a step in the environment using the given action index.
        :param action_idx: Index of the action to be taken.
        :return: Tuple of (observation, reward, done, truncated, info).
        """
        
        # Encode the action index into a one-hot vector
        a = torch.tensor(
            one_hot_encode(action_idx, self.action_space.n),
            dtype=torch.float32, device=self.dev).unsqueeze(0)

        # Imagine the next state using the world model's RSSM
        # Imagines returns (state, prior_logits, reward_logits, value_logits, cont_logits)
        with torch.no_grad():
            self.state, p_logits, r_logits, v_logits, c_logits = \
                self.wm.rssm.imagine(self.state, a)

            # Build probs over symlog-reward bins
            probs = torch.softmax(r_logits, dim=-1).squeeze(0)  # (return_bins,)

            # Construct equally spaced bin centers in [r_min, r_max]
            bins  = torch.linspace(self.r_min, self.r_max,
                                  steps=self.return_bins,
                                  device=self.dev)

            # Expected symlog reward
            r_syg   = torch.dot(probs, bins)

            # Invert symlog to real reward
            reward  = symexp(r_syg).item()

            # Continuation prediction
            cont    = torch.sigmoid(c_logits.squeeze(0)).item()

        self.turn += 1
        done = (self.turn >= self.max_T) or (cont < 0.5)

        # Build the observation vector for the controller
        obs_vec = _vec(self.state[1], self.state[0])  # (Z+H,)

        # Get the legality mask from the world model's legality network
        _, mask = self.get_valid_actions()  # (1st) mask from legality net

        info = {"action_mask": mask.astype(bool)}

        return obs_vec.cpu().numpy().astype(np.float32), reward, done, False, info

    def get_valid_actions(self):
        """
        Get the valid actions based on the current state and legality network.
        :return: Tuple of (logits, mask) where logits are the action probabilities and mask indicates valid actions.
        """
        with torch.no_grad():
            feat = torch.cat([self.state[1].squeeze(0),
                              self.state[0].squeeze(0)], -1)
            logits = self.wm.legality_net(feat.unsqueeze(0))
            mask   = (logits > 0).to(torch.int8).squeeze(0).cpu().numpy()
        return None, mask
    
    def render(self, *args, **kwargs): 
        return None
    
    def close(self): 
        self.realenv.close()


def make_env(
    world_model, real_env, device,
    cont_indices=None, disc_indices=None,
    observation_shape=None, max_steps=60
    ):
    """
    Helper function for creating a DreamHearthstoneEnv instance with the given parameters.
    :param world_model: The world model used for simulating future states.
    :param real_env: The real Hearthstone environment used for initial observation.
    :param device: The device to use for computations (CPU or GPU).
    :param cont_indices: Indices of continuous features in the observation space.
    :param disc_indices: Indices of discrete features in the observation space.
    :param observation_shape: Shape of the observation space.
    :param max_steps: The maximum number of steps in an episode.
    :return: A DreamHearthstoneEnv instance.
    """

    # Create the dream environment
    env = DreamHearthstoneEnv(
        world_model, real_env,
        device=device, max_steps=max_steps,
        cont_indices=cont_indices,
        disc_indices=disc_indices,
        observation_shape=observation_shape,
        return_bins=world_model.rssm.return_bins,
        )

    # Maskable-SB3 wrapper
    env = ActionMasker(env, mask_fn)
    return env
