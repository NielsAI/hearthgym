import random
import numpy as np

from agents.HearthstoneAgent import HearthstoneAgent
from env.hearthstone.HearthGym import HearthstoneEnv

class RandomAgent(HearthstoneAgent):
    """
    Random agent implementation that selects actions uniformly at random.
    """
    def __init__(self):
        super().__init__()

    def act(self, observation, valid_actions: list = None, action_mask = None, env: HearthstoneEnv = None):
        """
        Randomly selects an action from the action space.

        :param observation: The current observation from the environment.
        :param valid_actions: List of valid actions (not used in this implementation).
        :param action_mask: Mask for valid actions (not used in this implementation).
        :param env: The environment (not used in this implementation).
        :return: The selected action.
        """
                
        # Sample one of the valid actions
        if valid_actions:
            return random.choice(valid_actions)
        else:
            # If no valid actions are available, return a random action
            # Action space is multi discrete, so we need to sample each component from self.action_space
            return [np.random.randint(0, space.n) for space in self.action_space.spaces]