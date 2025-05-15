# env/hearthstone/HearthstoneVsAgentEnv.py
import pandas as pd

import random 
import gymnasium

from agents.RandomAgent import RandomAgent  # Import your fixed opponent agent

class HearthVsAgentEnv(gymnasium.Env):
    """
    Wrapper for the HearthGym environment to play against a fixed opponent agent.
    Commonly used for RL model training (e.g., PPO) and evaluation.
    
    :param class1: The class of the PPO agent (e.g., "all" for random).
    :param class2: The class of the opponent agent (e.g., "all" for random).
    :param class_options: Dictionary of class options.
    :param card_data: DataFrame containing card data.
    :param final_reward_mode: Mode for final reward calculation.
    :param incremental_reward_mode: Mode for incremental reward calculation.
    :param embedded: Boolean indicating if the environment is embedded.
    :param deck_include: Boolean indicating if deck information is included.
    :param deck_include_v2: Boolean indicating if deck information is included (v2).
    :param opponent_agent: The fixed opponent agent (default is RandomAgent).
    :param deck1: The deck of the PPO agent (e.g., "all" for random).
    :param deck2: The deck of the opponent agent (e.g., "all" for random).
    :param deck_metadata: DataFrame containing deck metadata.
    :param all_decks: DataFrame containing all decks.
    :param mirror_matches: Boolean indicating if mirror matches are allowed.
    """
    def __init__(
        self,  
        class1: int = None, 
        class2: int | str = None, 
        class_options: list = None,
        card_data: pd.DataFrame = None, 
        final_reward_mode: int = None, 
        incremental_reward_mode: int = None, 
        embedded: bool = False,
        deck_include: bool = False,
        deck_include_v2: bool = False,
        opponent_agent: RandomAgent = None,
        deck1: list | str = None,
        deck2: list | str = None,
        deck_metadata: pd.DataFrame = None,
        all_decks: pd.DataFrame = None,
        mirror_matches: bool = False,
        ):
        
        
        super().__init__()
        
        self.PPO_index = 0
        self.opponent_index = 1
        self.class_options = class_options
        self.deck_metadata = deck_metadata
        self.all_decks = all_decks
        self.mirror_matches = mirror_matches
        
        self.class1 = class1
        self.class2 = class2
        self.deck1 = deck1
        self.deck2 = deck2
        
        # If class1 is "all", then the agent will play with a random class
        if self.class1 is not None and self.class1 == "all":
            class1_selected = random.choice(list(self.class_options.keys()))
        else:
            class1_selected = self.class1
            
        if self.mirror_matches:
            class2_selected = class1_selected
        else:
            # If class2 is "all", then the opponent will be a RandomAgent with a random class
            if self.class2 is not None and self.class2 == "all":
                class2_selected = random.choice(list(self.class_options.keys()))
            else:
                class2_selected = self.class2
            
        # If deck1 is "all", then the agent will play with a random deck
        if self.deck1 is not None and type(self.deck1) == str:
            deck_id1 = random.choice(deck_metadata[deck_metadata['hero_class'] == class_options[class1_selected]]["deck_id"].values)
            deck1_selected = all_decks[all_decks['deck_id'] == deck_id1]['card_id'].values
        else:
            deck1_selected = self.deck1
            
        if self.mirror_matches:
            deck2_selected = deck1_selected
        else:
            # If deck2 is "all", then the opponent will be a RandomAgent with a random deck
            if self.deck2 is not None and type(self.deck2) == str:
                deck_id2 = random.choice(deck_metadata[deck_metadata['hero_class'] == class_options[class2_selected]]["deck_id"].values)
                deck2_selected = all_decks[all_decks['deck_id'] == deck_id2]['card_id'].values
            else:
                deck2_selected = self.deck2        
        
        print(f"Class1: {class1_selected}")
        print(f"Deck1: {deck1}")
        
        print(f"Class2: {class2_selected}")
        print(f"Deck2: {deck2}")
        
        
        # Initialize the base Hearthstone environment
        self.hs_env = gymnasium.make(
            id = "hearthstone_env/HearthGym-v0",
            class1=class1_selected, 
            class2=class2_selected, 
            class_options=self.class_options,
            clone=None, 
            card_data=card_data, 
            final_reward_mode=final_reward_mode, 
            incremental_reward_mode=incremental_reward_mode, 
            embedded=embedded,
            deck_include=deck_include,
            deck_include_v2=deck_include_v2,
            deck1=deck1_selected,
            deck2=deck2_selected,
            ).unwrapped
        
        # Initialize the fixed opponent agent
        self.opponent_agent = opponent_agent if opponent_agent is not None else RandomAgent()
        
        # Define the observation and action spaces
        self.observation_space = self.hs_env.observation_space
        self.action_space = self.hs_env.action_space  # MultiDiscrete or Dict as defined
        
        # Determine which player is the PPO agent (e.g., Player 1)
        self.ppo_player = f"Player{self.PPO_index + 1}"
        self.opponent_player = self.hs_env.game.players[self.opponent_index]  # Player 2 is opponent
        
        # Keep track of the current player
        self.current_player = self.hs_env.current_player        
        
    def reset(self, **kwargs):
        """ Reset the environment and initialize player turns. """
        # Reset the Hearthstone environment
        # If class1 is "all", then the agent will play with a random class
        if self.class1 is not None and self.class2 == "all":
            class1_selected = random.choice(list(self.class_options.keys()))
        else:
            class1_selected = self.class2
        
        if self.mirror_matches:
            class2_selected = class1_selected
        else:
            # If class2 is "all", then the opponent will be a RandomAgent with a random class
            if self.class2 is not None and self.class2 == "all":
                class2_selected = random.choice(list(self.class_options.keys()))
            else:
                class2_selected = self.class2
            
        self.deck_mapping = {}
        for hero_class in self.class_options.values():
            decks_for_class = self.deck_metadata[self.deck_metadata['hero_class'] == hero_class]["deck_id"].values
            # Cache the deck lists for each deck id for quick lookup
            deck_lists = {deck_id: self.all_decks[self.all_decks['deck_id'] == deck_id]['card_id'].values
                        for deck_id in decks_for_class}
            self.deck_mapping[hero_class] = {
                "deck_ids": decks_for_class,
                "deck_lists": deck_lists
            }
            
        # For agent deck selection:
        if self.deck1 is not None and isinstance(self.deck1, str) and self.deck1 == "all":
            # Select a random deck using precomputed mapping
            decks_for_class = self.deck_mapping[self.class_options[class1_selected]]
            deck_ids = decks_for_class["deck_ids"]
            deck_id1 = random.choice(deck_ids)
            deck1_selected = decks_for_class["deck_lists"][deck_id1]
        else:
            deck1_selected = self.deck1
            
        if self.mirror_matches:
            deck2_selected = deck1_selected
        else:
            # For opponent deck selection:
            if self.deck2 is not None and isinstance(self.deck2, str) and self.deck2 == "all":
                decks_for_class = self.deck_mapping[self.class_options[class2_selected]]
                deck_ids = decks_for_class["deck_ids"]
                deck_id2 = random.choice(deck_ids)
                deck2_selected = decks_for_class["deck_lists"][deck_id2]
            else:
                deck2_selected = self.deck2
                
        # Update classes in hs_env
        self.hs_env.class1 = class1_selected
        self.hs_env.class2 = class2_selected
         
        # Update decks in hs_env
        self.hs_env.deck1 = deck1_selected
        self.hs_env.deck2 = deck2_selected
        
        # Reset the Hearthstone environment
        observation, info = self.hs_env.reset()
        
        # Store action mask in info
        _, info["action_mask"] = self.hs_env.get_valid_actions()
        
        self.current_player = self.hs_env.current_player
                
        return observation, info
    
    def get_valid_actions(self):
        """
        Get the valid actions for the current player.
        
        :return: Tuple (valid_actions, action_mask)
        """
        valid_actions, action_mask = self.hs_env.get_valid_actions()
        return valid_actions, action_mask
    
    def step(self, action):
        """
        Execute an action for the PPO agent or the fixed opponent.
        
        :param action: The action taken by the PPO agent (ignored if opponent's turn)
        :return: Tuple (observation, reward, done, info)
        """
        # Update the current player
        self.current_player = self.hs_env.current_player
        current_player = self.current_player
        
        if current_player.name == self.ppo_player:
            # PPO agent's turn
            obs, reward, done, truncated, info = self.hs_env.step(action=action)
            
            if not info["valid_action"]:
                reward = -0.1  # Penalize invalid actions (Only for RNN PPO)                
        else:
            # Opponent agent's turn
            observation = self.hs_env._get_obs()
            valid_actions, action_mask = self.hs_env.get_valid_actions()
            opponent_action = self.opponent_agent.act(observation, valid_actions=valid_actions, action_mask=action_mask, env=self.hs_env)
            obs, reward, done, truncated, info = self.hs_env.step(opponent_action)
            
        
            # If the game is done, set the reward to the final reward
            if not done:
                reward = 0.0  # No reward for opponent's actions
            else:
                # Multiply the reward by -1 to reflect the opponent's perspective
                reward*= -1.0
                        
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        :param mode: The mode for rendering (default is 'human').
        :return: The rendered output.
        """
        return self.hs_env.render(mode)
    
    def close(self):
        return self.hs_env.close()

    def flat_index_to_action_dict(self, idx) -> dict:
        """ Convert a flat index back into an action dictionary. """
        choose_index = idx % self.MAX_GENERIC_CHOICES
        idx //= self.MAX_GENERIC_CHOICES
        discover_index = idx % self.MAX_DISCOVER_OPTIONS
        idx //= self.MAX_DISCOVER_OPTIONS
        target_index = idx % self.MAX_TOTAL_CHARACTERS
        idx //= self.MAX_TOTAL_CHARACTERS
        attacker_index = idx % self.MAX_NUM_CHARACTERS
        idx //= self.MAX_NUM_CHARACTERS
        card_index = idx % self.MAX_HAND_SIZE
        idx //= self.MAX_HAND_SIZE
        action_type = idx
        return {
            'action_type': int(action_type),
            'card_index': int(card_index),
            'attacker_index': int(attacker_index),
            'target_index': int(target_index),
            'discover_index': int(discover_index),
            'choose_index': int(choose_index)
        }
        
    def _action_to_index(self, action: dict) -> int:
        """Convert an action dict to a flat index using precomputed multipliers."""
        # Precompute multipliers
        m1 = self.MAX_HAND_SIZE
        m2 = m1 * self.MAX_NUM_CHARACTERS
        m3 = m2 * self.MAX_TOTAL_CHARACTERS
        m4 = m3 * self.MAX_DISCOVER_OPTIONS
        m5 = m4 * self.MAX_GENERIC_CHOICES
        
        # Flatten the action in the given order
        idx = action['action_type']
        idx = idx * self.MAX_HAND_SIZE + action['card_index']
        idx = idx * self.MAX_NUM_CHARACTERS + action['attacker_index']
        idx = idx * self.MAX_TOTAL_CHARACTERS + action['target_index']
        idx = idx * self.MAX_DISCOVER_OPTIONS + action['discover_index']
        idx = idx * self.MAX_GENERIC_CHOICES + action['choose_index']
        return idx
        
    def _update_opponent(self, opponent_agent):
        """
        Update the opponent agent.
        
        :param opponent_agent: The new opponent agent.
        """
        self.opponent_agent = opponent_agent