from env.hearthstone.HearthGym import HearthstoneEnv
from agents.HearthstoneAgent import HearthstoneAgent

from models.PPO.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO
from models.EncodedPPO.MultiHeadAutoEncoder import MultiHeadAutoEncoder
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO
import torch

import numpy as np

class EncodedPPOAgent(HearthstoneAgent):
    """
    Agent implementation that uses Proximal Policy Optimization (PPO) to learn a policy.
    """
    def __init__(self, encoder_path: str, model_path=None, model_type="Mask", cont_indices=None, disc_indices=None):
        """
        Initialize the PPO agent.
        
        :param model_path (str): The path to the trained model.
        :param device (str): The device to use for inference.
        """
        
        super().__init__()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # ---------- load world-model --------------------------------------------------
        enc_ckpt = torch.load(encoder_path, map_location=self.device, weights_only=True)
        self.autoencoder = MultiHeadAutoEncoder(
            cont_input_dim = enc_ckpt["cont_input_dim"],
            disc_input_dim = enc_ckpt["disc_input_dim"],
            latent_dim     = enc_ckpt["latent_dim"],
            cont_hidden_dim=128, 
            disc_hidden_dim=32).to(self.device)
        self.autoencoder.load_state_dict(enc_ckpt["autoencoder_state_dict"])
        self.autoencoder.eval()
        
        if model_path is None:
            model_path = "src\\models\\PPO\\trained\\ppo_hs_agent.zip"
            
        self.model_type = model_type
        
        if self.model_type == "MaskRNN":
            # Load the model
            self.model = RecurrentMaskablePPO.load(model_path, device=self.device)
        elif self.model_type == "RNN":
            self.model = RecurrentPPO.load(model_path, device=self.device)
        elif self.model_type == "Mask":
            self.model = MaskablePPO.load(model_path, device=self.device)
        else:
            self.model = PPO.load(model_path, device=self.device)
            
        self.cont_idx = cont_indices
        self.disc_idx = disc_indices
        
    def _encode(self, obs_np):
        obs = torch.as_tensor(
            obs_np, dtype=torch.float32,
            device=self.device).unsqueeze(0)
        latent_state   = self.autoencoder.encode(obs, self.cont_idx, self.disc_idx)
        return latent_state                        
        
    def act(self, observation, valid_actions: list = None, action_mask = None, env: HearthstoneEnv = None):
        """
        Decide on an action based on the given observation.
        Should be implemented by subclasses.

        :param observation (np.ndarray): Current state observation.
        :param valid_actions (tuple): Tuple containing the list of valid actions and the action mask.
        :param env (HearthstoneEnv): The environment instance.
        :return: dict: An action dictionary compatible with the environment.
        """
              
        # Encode the observation
        latent_state = self._encode(observation)
        
        # Convert the latent state to a numpy array
        latent_state = latent_state.detach().cpu().numpy().astype(np.float32)
              
        if self.model_type in ["Mask", "MaskRNN"]:  
            # Sample one of the valid actions
            action, _ = self.model.predict(latent_state, deterministic=False, action_masks=action_mask)
        else:
            action, _ = self.model.predict(latent_state, deterministic=False)
        
        return action
    
    class CustomScore:
        pass