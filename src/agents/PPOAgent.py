from env.hearthstone.HearthGym import HearthstoneEnv
from agents.HearthstoneAgent import HearthstoneAgent

from models.PPO.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import PPO
import torch

class PPOAgent(HearthstoneAgent):
    """
    Agent implementation that uses Proximal Policy Optimization (PPO) to learn a policy.
    """
    def __init__(self, model_path=None, model_type="Mask"):
        """
        Initialize the PPO agent.
        
        :param model_path (str): The path to the trained model.
        :param device (str): The device to use for inference.
        """
        
        super().__init__()
        
        if model_path is None:
            model_path = "src\\models\\PPO\\trained\\ppo_hs_agent.zip"
            
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        self.model_type = model_type
        
        if self.model_type == "MaskRNN":
            # Load the model
            self.model = RecurrentMaskablePPO.load(model_path, device=device)
        elif self.model_type == "RNN":
            self.model = RecurrentPPO.load(model_path, device=device)
        elif self.model_type == "Mask":
            self.model = MaskablePPO.load(model_path, device=device)
        else:
            self.model = PPO.load(model_path, device=device)
        
    def act(self, observation, valid_actions: list = None, action_mask = None, env: HearthstoneEnv = None):
        """
        Decide on an action based on the given observation.
        Should be implemented by subclasses.

        :param observation (np.ndarray): Current state observation.
        :param valid_actions (tuple): Tuple containing the list of valid actions and the action mask.
        :param env (HearthstoneEnv): The environment instance.
        :return: dict: An action dictionary compatible with the environment.
        """
              
        if self.model_type in ["Mask", "MaskRNN"]:  
            # Sample one of the valid actions
            action, _ = self.model.predict(observation, deterministic=False, action_masks=action_mask)
        else:
            action, _ = self.model.predict(observation, deterministic=False)
        
        return action
    
    class CustomScore:
        pass