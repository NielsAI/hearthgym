from env.hearthstone.HearthGym import HearthstoneEnv
from fireplace.game import Game
import copy 

import pandas as pd
import numpy as np    
    
def init_worker():
    from fireplace import cards
    cards.db.initialize()
    
class HearthstoneAgent:
    """
    Base class for Hearthstone agents. All custom agents should inherit from this class.
    """
    def __init__(self):
        self.action_space = None

    def load_action_space(self, action_space: dict) -> None:
        """
        Load the action space for the agent.
        
        :param action_space (dict): The action space dictionary.
        """
        
        self.action_space = action_space

    def act(self, observation, valid_actions: list = None, action_mask = None, env: HearthstoneEnv = None) -> dict | np.ndarray:
        """
        Decide on an action based on the given observation.
        Should be implemented by subclasses.

        :param observation (np.ndarray): Current state observation.
        :param valid_actions (list): List of valid actions.
        :param action_mask (np.ndarray): Mask for valid actions.
        :param env (HearthstoneEnv): The environment instance.
        :return: dict or list: An action dictionary / list compatible with the environment.
        """
        raise NotImplementedError("The 'act' method must be implemented by subclasses.")
    
    def simulate_action(self, action: dict, class1: str, class2: str, game_state: Game, card_data: pd.DataFrame) -> tuple:
        """
        Simulate the effect of an action in a separate environment instance.
        
        :param action (dict): The action to simulate.
        :param class1 (str): The class of the first player.
        :param class2 (str): The class of the second player.
        :param game_state (Game): The game to clone.
        :param card_data (pd.DataFrame): The card data to use for the simulation.
        :return: tuple: The action and the environment instance.
        """
                
        env_copy = HearthstoneEnv(class1=class1, class2=class2, clone=game_state, card_data=card_data, final_reward_mode=0, incremental_reward_mode=1)
        _ = env_copy.step(action)
        
        return (action, env_copy)
    
    # NON-PARALLELIZED VERSION
    def simulate(self, action_list: list, env: HearthstoneEnv) -> list:
        """
        Simulate the effect of a list of actions in parallel.
        
        :param action_list (list): The list of actions to simulate.
        :param env (HearthstoneEnv): The environment instance.
        :return: list: The list of tuples containing the action and the environment instance.
        """
            
        options = [self.simulate_action(action, env.class1, env.class2, copy.deepcopy(env.game), env.card_data) for action in action_list]
                        
        return options
    
    class CustomScore:
        pass