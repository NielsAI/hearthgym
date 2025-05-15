# agents/WorldModelAgent.py
import numpy as np, torch
from agents.HearthstoneAgent import HearthstoneAgent
from models.WorldModel.utils  import one_hot_encode, symlog, symexp
from models.WorldModel.RSSM   import RSSM
from models.WorldModel.MultiHeadAutoEncoder import MultiHeadAutoEncoder

# Stable-Baselines3 families
from models.PPO.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO
from models.WorldModel.TwoHotMaskable import TwoHotMaskablePPO
from env.hearthstone.HearthGym import HearthstoneEnv

def _vec(z_star: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Build controller feature vector = [ symexp(z*) , h ]   (Z+H,)
    Both inputs still have a batch-dim (1,⋯); squeeze it out.
    """
    z = symexp(z_star.squeeze(0))
    return torch.cat([z, h.squeeze(0)], 0)

class WorldModelAgent(HearthstoneAgent):
    """
    Wrapper that embeds a Dreamer-V3 world-model into a Hearthstone agent.
    At every real turn it
        1. encodes the env observation → z;
        2. **updates** the RSSM posterior with (prev_z, prev_a, z);
        3. feeds [symexp(z), h] to the PPO controller and returns an action.
    """

    def __init__(
        self,
        encoder_path:     str,
        rssm_path:        str,
        controller_path:  str,
        ppo_model_type:   str = "Mask",
        cont_indices=None,
        disc_indices=None
        ):
        super().__init__()
        
        # Set device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Load AutoEncoder
        enc_ckpt = torch.load(encoder_path, map_location=self.device, weights_only=True)
        self.autoencoder = MultiHeadAutoEncoder(
            cont_input_dim = enc_ckpt['cont_input_dim'],
            disc_input_dim = enc_ckpt['disc_input_dim'],
            latent_dim     = enc_ckpt['latent_dim'],
            cont_hidden_dim=enc_ckpt['cont_hidden_dim'],
            disc_hidden_dim=enc_ckpt['disc_hidden_dim']
            ).to(self.device)
        self.autoencoder.load_state_dict(enc_ckpt["autoencoder_state_dict"])
        self.autoencoder.eval()

        # Load RSSM
        rssm_ckpt = torch.load(rssm_path, map_location=self.device, weights_only=False)
        self.rssm = RSSM(
            cat_dim     = rssm_ckpt['cat_dim'],
            num_cats    = rssm_ckpt['num_cats'],
            hidden_dim  = rssm_ckpt['hidden_dim'],
            action_dim  = rssm_ckpt['action_dim'],
        ).to(self.device)
        self.rssm.load_state_dict(rssm_ckpt["state_dict"])
        self.rssm.eval()

        # Load PPO controller
        if ppo_model_type == "MaskRNN":
            self.ctrl = RecurrentMaskablePPO.load(controller_path, device=self.device)
        elif ppo_model_type == "RNN":
            self.ctrl = RecurrentPPO.load(controller_path, device=self.device)
        elif ppo_model_type == "Mask":
            self.ctrl = MaskablePPO.load(controller_path, device=self.device)
        elif ppo_model_type == "TwoHotMask":
            self.ctrl = TwoHotMaskablePPO.load(controller_path, device=self.device)
        else:
            self.ctrl = PPO.load(controller_path, device=self.device)

        # Store class properties
        self.cont_idx = cont_indices
        self.disc_idx = disc_indices
        self.state    = None                    
        self.prev_a   = torch.zeros(1, rssm_ckpt['action_dim'], device=self.device) 
        self.action_space = None                

    def load_action_space(self, space):
        # Store action space for later use
        self.action_space = space

    def _encode(self, obs_np):
        """Encode the observation into a latent space representation.
        :param obs_np: [B, D] full observation (numpy array).
        :return: [B, Z] latent space representation.
        """
        # Convert numpy array to PyTorch tensor
        obs = torch.as_tensor(
            obs_np, dtype=torch.float32,
            device=self.device).unsqueeze(0)
        
        # Encode the observation using the autoencoder
        z   = self.autoencoder.encode(obs, self.cont_idx, self.disc_idx)
        return symlog(z)

    def new_game(self, env):
        """Called by run_game right after env.reset().
        :param env: The environment object.
        """
        # Initialize the action space and previous action
        self.state  = self.rssm.init_state(1, self.device)
        self.prev_a = torch.zeros(1, env.action_space.n, device=self.device)

    def act(self, observation, valid_actions: list = None, action_mask = None, env: HearthstoneEnv = None) -> dict | np.ndarray:
        """Decide on an action based on the given observation.
        :param observation: Current state observation.
        :param valid_actions: List of valid actions.
        :param action_mask: Mask for valid actions.
        :param env: The environment instance.
        :return: An action dictionary / list compatible with the environment.
        """

        # Encode the observation into a latent space representation
        z_t = self._encode(observation)                     # (1, Z)

        # RSSM update: (prev_z, prev_a, z_t) → (z_t, h_t)
        self.state, p_logits, q_logits, r_log, v_log, c_log = self.rssm.observe(self.state, self.prev_a, z_t)

        # Vectorize the controller observation
        ctrl_obs = _vec(self.state[1], self.state[0])
        ctrl_obs = ctrl_obs.detach().cpu().numpy().astype(np.float32)

        # Predict the action using the controller model
        if isinstance(self.ctrl, (MaskablePPO, RecurrentMaskablePPO)):
            action, _ = self.ctrl.predict(ctrl_obs,
                                           deterministic=False,
                                           action_masks=action_mask)
        else:
            action, _ = self.ctrl.predict(ctrl_obs, deterministic=False)

        # Store one-hot code of previous action in the previous action variable
        self.prev_a = torch.as_tensor(
            one_hot_encode(action, env.action_space.n),
            dtype=torch.float32, device=self.device).unsqueeze(0)

        return action
