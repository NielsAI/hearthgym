# Import Hearthstone modules (from fireplace)
from fireplace.game import Game
from hearthstone.enums import PlayState

# Import generic modules
import gymnasium
import random
from gymnasium import spaces
import pandas as pd
import numpy as np
from numba import njit

# Import functions from utils.py to reduce duplicate code
from env.hearthstone.utils import setup_game
from functions.reward_functions import calculate_intermediate_reward_numba

# Constants for the environment
MAX_HAND_SIZE = 10
MAX_BOARD_SIZE = 7
MAX_DECK_SIZE = 30
MAX_DECK_LIMIT = 60
MAX_NUM_MINIONS = MAX_BOARD_SIZE
MAX_NUM_CHARACTERS = MAX_BOARD_SIZE + 1  # +1 for hero
MAX_TOTAL_CHARACTERS = 2 * MAX_NUM_CHARACTERS  # Both players
MAX_ACTION_TYPES = 5  # End turn, Play card, Use hero power, Attack with minion, Attack with hero
MAX_DISCOVER_OPTIONS = 4
MAX_GENERIC_CHOICES = 4

# Stats: attack, health, max_health, can_attack, divine_shield, stealthed, frozen, silenced, windfury, mega_windfury, immune_while_attacking, has_inspire, has_battlecry, has_deathrattle, has_overkill, lifesteal
N_MINION_STATS = 16
N_CARD_CLASSES = 15

class HearthstoneEnv(gymnasium.Env):    
    """
    Class representing the HearthGym gymnasium-compatible environment for Hearthstone RL.
    
    :param class1: The class of player 1
    :param class2: The class of player 2
    :param class_options: The class options for the players
    :param clone: A cloned game object to continue
    :param card_data: The card data
    :param final_reward_mode: The final reward mode
    :param incremental_reward_mode: The incremental reward mode
    :param embedded: Whether to use embedded card names
    :param deck_include: Whether to include the deck in the state
    :param deck1: The deck for player 1
    :param deck2: The deck for player 2
    """
    def __init__(
        self, 
        class1: int = None, 
        class2: int = None, 
        class_options: dict = None,
        clone: Game = None, 
        card_data: pd.DataFrame = None,
        final_reward_mode: int = None, 
        incremental_reward_mode: int = None, 
        embedded: bool = False,
        deck_include: bool = False,
        deck_include_v2: bool = False,
        deck1: list = None,
        deck2: list = None,
        ) -> None:
        
        super(HearthstoneEnv, self).__init__()
        
        self.print_width = 75
        
        # Set various static variables for the environment
        self.MAX_HAND_SIZE = MAX_HAND_SIZE
        self.MAX_BOARD_SIZE = MAX_BOARD_SIZE
        self.MAX_DECK_SIZE = MAX_DECK_SIZE
        self.MAX_DECK_LIMIT = MAX_DECK_LIMIT
        self.MAX_NUM_MINIONS = MAX_BOARD_SIZE
        self.MAX_NUM_CHARACTERS = MAX_BOARD_SIZE + 1  # +1 for hero
        self.MAX_TOTAL_CHARACTERS = 2 * MAX_NUM_CHARACTERS  # Both players
        self.MAX_ACTION_TYPES = MAX_ACTION_TYPES  # End turn, Play card, Use hero power, Attack with minion, Attack with hero
        self.MAX_DISCOVER_OPTIONS = MAX_DISCOVER_OPTIONS
        self.MAX_GENERIC_CHOICES = MAX_GENERIC_CHOICES
        
        # Set the state space modes
        self.embedded = embedded
        self.deck_include = deck_include
        self.deck_include_v2 = deck_include_v2
        
        # Save the classes and decks
        self.class1 = class1
        self.class2 = class2
        self.class_options = class_options
        
        self.deck1 = deck1
        self.deck2 = deck2
        
        # Store the card data and reward modes
        self.card_data = card_data
        self.final_reward_mode = final_reward_mode
        self.incremental_reward_mode = incremental_reward_mode        
        
        # If a clone is provided, use it to initialize the game, otherwise create a new game
        if clone:            
            self.game = clone
            self.current_player = self.game.current_player
        else:

            # Initialize the game using setup_game from utils.py
            self.game, self.card_collections = setup_game(
                class1=self.class1, 
                class2=self.class2,
                deck1=self.deck1,
                deck2=self.deck2
                )
            
            # Generate the hands for both players based on the generated decks
            for player in self.game.players:
                mull_count = random.randint(0, len(player.choice.cards))
                cards_to_mulligan = random.sample(player.choice.cards, mull_count)
                player.choice.choose(*cards_to_mulligan)
            
            self.current_player = self.game.current_player
        
        self.card_id_to_index = {cid: idx for idx, cid in enumerate(self.card_data['id'])}
        
        self.num_flat_actions = (
            MAX_ACTION_TYPES
            * MAX_HAND_SIZE
            * MAX_NUM_CHARACTERS
            * MAX_TOTAL_CHARACTERS
            * MAX_DISCOVER_OPTIONS
            * MAX_GENERIC_CHOICES
        )
        self.action_space = spaces.Discrete(self.num_flat_actions)
        
        # Define the state space
        state_space_size = (
            N_CARD_CLASSES + # Player class encoding (one-hot encoding)
            1 + # Player's mana (continuous)
            1 + # Player's max mana (continuous)
            1 + # Player's hero health (continuous)
            1 + # Player's hero attack (continuous)
            1 + # Player's hero armor (continuous)
            1 + # Player's choice (boolean)
            1 + # Opponent's hero health (continuous)
            1 + # Opponent's mana (continuous)
            1 + # Opponent's max mana (continuous)
            1 + # Opponent's hand size (continuous)
            1 + # Turn number (continuous)
            1 + # Hero power availability (boolean)
            1 + # Hero weapon attack (continuous)
            1 + # Hero weapon durability (continuous)
            MAX_BOARD_SIZE * N_MINION_STATS + # Player's board minion stats (mix of continuous and boolean)
            MAX_BOARD_SIZE * N_MINION_STATS # Opponent's board minion stats (mix of continuous and boolean)
        )
        
        # If deck_include is True, add the largest collection size to the state space
        if self.deck_include:
            # Store player beginning decks
            self.player1_deck = self.game.players[0].deck
            self.player2_deck = self.game.players[1].deck
                    
            # Add the largest collection size to the state space (1016 for now from Rogue/Warlock)
            state_space_size += 1016
        
        if self.embedded or self.deck_include_v2:
            self.cards_embedded = np.load("src/data/card_embeddings.npz")
            embedding_size = self.cards_embedded['embeddings'].shape[1]
            
        if self.embedded:            
            # Add the embedding size to the state space for the hand and board cardsn (for both players)
            state_space_size += 3 * embedding_size # hand, board, opponent board
            
        if self.deck_include_v2:
            # Add the deck size to the state space for the current player
            state_space_size += embedding_size # deck embedding
        
        self.observation_space = spaces.Box(
            low=0,
            high=99999,
            shape=(state_space_size,),
            dtype=np.float32
        )
                
        # Initialize previous state metrics for reward calculation
        self.prev_hero_healths = [player.hero.health for player in self.game.players]
        self.prev_minions = [len(player.field) for player in self.game.players]
        self.prev_mana = [player.mana for player in self.game.players]
        self.prev_hand_sizes = [len(player.hand) for player in self.game.players]
        
        # Build the observation indices for faster access
        idx = 0
        self.obs_indices = {}
        self.obs_indices['player_class_start'] = idx; idx += N_CARD_CLASSES
        self.obs_indices['player_mana'] = idx; idx += 1
        self.obs_indices['max_mana'] = idx; idx += 1
        self.obs_indices['hero_health'] = idx; idx += 1
        self.obs_indices['hero_attack'] = idx; idx += 1
        self.obs_indices['hero_armor'] = idx; idx += 1
        self.obs_indices['has_choice'] = idx; idx += 1
        self.obs_indices['opponent_health'] = idx; idx += 1
        self.obs_indices['board_stats_start'] = idx; idx += MAX_BOARD_SIZE * N_MINION_STATS
        self.obs_indices['opp_board_stats_start'] = idx; idx += MAX_BOARD_SIZE * N_MINION_STATS
        self.obs_indices['opponent_mana'] = idx; idx += 1
        self.obs_indices['opponent_max_mana'] = idx; idx += 1
        self.obs_indices['opponent_hand_size'] = idx; idx += 1
        self.obs_indices['turn_number'] = idx; idx += 1
        self.obs_indices['hero_power_available'] = idx; idx += 1
        self.obs_indices['hero_weapon_attack'] = idx; idx += 1
        self.obs_indices['hero_weapon_durability'] = idx; idx += 1
        
        # If deck_include is True, add the largest collection size to the state space
        if self.deck_include:
            self.obs_indices['deck_start'] = idx; idx += 1016  # assumed fixed collection size
            
        # If embedded is True, add the embedding size to the state space for the hand and board cards
        if self.embedded:
            self.cards_embedded = np.load("src/data/card_embeddings.npz")
            self.embedding_size = self.cards_embedded['embeddings'].shape[1]
            self.obs_indices['embedding_hand_start'] = idx; idx += self.embedding_size
            self.obs_indices['embedding_board_start'] = idx; idx += self.embedding_size
            self.obs_indices['embedding_opp_board_start'] = idx; idx += self.embedding_size
            
        if self.deck_include_v2:
            self.obs_indices['deck_start_v2'] = idx; idx += self.embedding_size
            
        self.done = False
    
    def step(self, action: dict | np.ndarray | np.int64) -> tuple:
        """
        Perform an action in the environment
        
        :param action: The action to perform
        :return: The observation, reward, whether the episode has ended, and additional information
        """
                
        # While training, PPO will pass the action as np.int64
        # While playing, PPO will pass the action as np.ndarray
        if type(action) != dict:
            action_dict = self.flat_index_to_action_dict(idx=action.copy())
        else:
            action_dict = action
            
        # Extract the action parameters
        action_type = action_dict['action_type'] # 0: End turn, 1: Play card, 2: Use hero power, 3: Attack with minion, 4: Attack with hero
        card_index = action_dict['card_index'] # Index of the card in the player's hand
        attacker_index = action_dict['attacker_index'] # Index of the attacking minion in the player's field
        target_index = action_dict['target_index'] # Index of the target minion or hero
        discover_index = action_dict['discover_index'] # Index of the chosen card from a Discover effect
        choose_index = action_dict['choose_index'] # Index of the chosen card from a choice effect (Hero Power or card)
                
        # Info dictionary to store additional information
        info = {}
        info["valid_action"] = True
                
        current_player = self.current_player
                
        try:
            # Just to filter out actions where the target is invalid
            can_play = True
            
            # Check if the player has a choice to make, and if so, make the choice instead of performing other actions
            if current_player.choice:                
                self._resolve_choices(discover_index=discover_index)
            else:
                if action_type == 0:
                    # End turn
                    self.game.end_turn()
                    
                elif action_type == 1:
                    # Play card
                    if card_index < len(current_player.hand):
                        card = current_player.hand[card_index]
                        
                        # Check if the card is playable
                        if card.is_playable():
                            target = None
                            
                            # If the card requires a choice, use the chosen card index
                            if card.must_choose_one:
                                while card.must_choose_one:
                                    card = card.choose_cards[choose_index]
                                
                                # Check if the chosen card is playable (can be minion that first goes to hand)
                                if not card.is_playable():
                                    can_play = False                                
                                    
                            # If the card requires a target, use target_index to determine the target from card.targets
                            if card.requires_target():
                                if target_index < len(card.targets):
                                    target = card.targets[target_index]
                                else:
                                    raise ValueError("Invalid target index for card")
                                    
                            if can_play:
                                # If the action is still valid, play the card
                                card.play(target=target)
                               
                            
                            self._resolve_choices(discover_index=discover_index) 
                            
                        else:
                            raise ValueError("Card is not playable")
                    else:
                        raise ValueError("Invalid card index")
                            
                elif action_type == 2:
                    # Use hero power
                    heropower = current_player.hero.power
                    
                    # Check if the hero power is usable
                    if heropower.is_usable():
                        choose = None
                        target = None
                        
                        # If the hero power requires a choice, use the chosen card index
                        if heropower.must_choose_one:
                            choose = heropower.choose_cards[choose_index]
                            
                        # If the hero power requires a target, use target_index to determine the target from heropower.targets
                        if heropower.requires_target():
                            if target_index < len(heropower.targets):
                                target = heropower.targets[target_index]
                            else:
                                raise ValueError("Invalid target index for hero power")
                                
                        # If the action is still valid, use the hero power
                        heropower.use(target=target, choose=choose)
                        
                        self._resolve_choices(discover_index=discover_index)
                        
                    else:
                        raise ValueError("Hero power is not usable")
                        
                elif action_type == 3:
                    # Attack with minion
                    if attacker_index < len(current_player.field):
                        minion = current_player.field[attacker_index]
                        
                        # Check if the minion can attack
                        if minion.can_attack():
                            # If the target is valid (within the bounds of minion.targets), attack the target
                            if target_index < len(minion.targets):
                                minion.attack(minion.targets[target_index])
                                
                                self._resolve_choices(discover_index=discover_index)
                            else:
                                raise ValueError("Invalid target index")
                        else:
                            raise ValueError("Minion cannot attack")
                    else:
                        raise ValueError("Invalid attacker index")
                                
                elif action_type == 4:
                    # Attack with hero
                    hero = current_player.hero
                    
                    # Check if the hero can attack
                    if hero.can_attack():
                        if target_index < len(hero.targets):
                            hero.attack(hero.targets[target_index])
                            
                            self._resolve_choices(discover_index=discover_index)
                        else:
                            raise ValueError("Invalid target index")
                    else:
                        raise ValueError("Hero cannot attack")
                
                else:
                    raise ValueError(f"Invalid action type: {action_type}")
                        
        except Exception as e:
            # If the game has ended, ignore the exception
            if not self.game.ended:
                print("Error: %s" % (e))
                # print(self.render())
                print(f"Invalid action by {current_player.name} - {current_player.hero.name_enUS}")
                info["valid_action"] = False  
        
        if not self.game.ended:  
            try: 
                # Do another clean_board to ensure minions are correctly removed
                self._clean_board()   
            except Exception as e:
                if not "ended" in str(e):
                    print("Unable to clean board")
                    print("Error: %s" % (e))
        
        # Get the neccessary information for the next step
        self.done = self.game.ended
        
        # Double check if the game has ended by checking player health
        # Does not matter if the game has ended, as the reward will be calculated based on the player's playstate
        if current_player.hero.health <= 0 or current_player.opponent.hero.health <= 0:
            self.done = True        
        
        # Setup the reward
        reward = 0.0
        
        if info["valid_action"]:
            intermediate_reward = self._calculate_intermediate_reward()
            reward += intermediate_reward
        else:
            reward = -1.0
            
        if self.done:
            reward, player_result = self._calculate_final_reward()
            info["player_result"] = player_result
        else:
            info["player_result"] = "not_done"
            
        # Update the previous state metrics for the next step
        self._update_previous_state_metrics()
        
        # Update the current player and get the observation for the next step
        self.current_player = self.game.current_player
        observation = self._get_obs()
        
        return observation, reward, self.done, None, info
    
    def reset(self, **kwargs) -> tuple:
        """
        Reset the environment
        
        :return: The initial observation and additional information
        """
        
        # Initialize the game using setup_game from utils.py
        self.game, self.card_collections = setup_game(
                class1=self.class1, 
                class2=self.class2,
                deck1=self.deck1,
                deck2=self.deck2
                )
        
        # Generate the hands for both players based on the generated decks
        for player in self.game.players:
            mull_count = random.randint(0, len(player.choice.cards))
            cards_to_mulligan = random.sample(player.choice.cards, mull_count)
            player.choice.choose(*cards_to_mulligan)
        
        self.current_player = self.game.current_player
        
        self.done = False
        
        # Info empty for now
        info = {}
        
        return self._get_obs(), info
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment
        
        :param mode: The mode to render the environment (human or rgb_array)
        :return: None
        """
        
        print("\nPlayer Info:\n")
        
        # Print the class of the current player
        
        
        print("=" * self.print_width)
        print("Hero:\n")
        print(f"{self.current_player.hero.name_enUS} (Health: {self.current_player.hero.health}, Attack: {self.current_player.hero.atk}, Armor: {self.current_player.hero.armor})")
        print("-" * self.print_width)
        
        print("Mana:", self.current_player.mana, "/", self.current_player.max_mana)
        
        print("-" * self.print_width)
        print("Hand:")
        for idx, card in enumerate(self.current_player.hand):
            print(f"{idx}: {card.name_enUS} ({card.cost} Mana) '{card.description.replace('<b>', '').replace('</b>', '')}'")
        
        print("\nBoards:\n")
        print("=" * self.print_width)
        print("Your Board:")
        if len(self.current_player.field) == 0:
            print("No minions on your board")
        else:
            for idx, minion in enumerate(self.current_player.field):
                print(f"{idx}: {minion.name_enUS} (Attack: {minion.atk}, Health: {minion.health}/{minion.max_health})")
        opponent = self.current_player.opponent
        
        print("=" * self.print_width)
        print("Opponent's Board:")
        if len(opponent.field) == 0:
            print("No minions on opponent's board")
        else:
            for idx, minion in enumerate(opponent.field):
                print(f"{idx}: {minion.name_enUS} (Attack: {minion.atk}, Health: {minion.health}/{minion.max_health})")
        print("=" * self.print_width)
    
        print("\nOpponent Hero:\n")
        print(f"{opponent.hero.name_enUS} (Health: {opponent.hero.health}, Attack: {opponent.hero.atk}, Armor: {opponent.hero.armor})")
        print("=" * self.print_width)
        
    
    def render_actions(self, valid_actions: list) -> None:
        """
        Render the valid actions for the human player
        
        :param valid_actions: The valid actions
        :return: None
        """
                
        # Convert the valid actions to a user-friendly format
        for i, action in enumerate(valid_actions):
            print_string = self._action_to_string(index = i, action = action)
            print(print_string)
            
        print("=" * self.print_width)
    
    def _action_to_string(self, index: int, action: dict) -> str:
        """
        Convert the action to a string
        
        :param action: The action to convert to a string
        :return: The action as a string
        """
        
        action_types = {
            0: "End turn",
            1: "Play card",
            2: "Use hero power",
            3: "Attack with minion",
            4: "Attack with hero"
        }
        
        # Extract the action parameters
        action_type = action_types[action['action_type']]
        
        # Extract the card that will be played
        card_info = self.current_player.hand[action['card_index']] if action['action_type'] == 1 else None
        card_index = action['card_index']
        card_name = card_info.name_enUS if card_info else None
        card_description = card_info.description.replace("<b>", "").replace("</b>", "") if card_info else None
        card_cost = card_info.cost if card_info else None
        
        # Combine card information into a single string
        card = f"{card_index} {card_name} ({card_cost} Mana) '{card_description}'" if card_name else None
        
        attacker = None
        target = None
        # Extract the target minion or hero based on whether the action is an attack / hero power, or card
        if action['action_type'] == 1: # Play card
            if card_info.requires_target():
                target_info = card_info.targets[action['target_index']]
                target = f"{action['target_index']} {target_info.name_enUS} (Attack: {target_info.atk} - Health: {target_info.health}/{target_info.max_health})"
            attacker = None
            
        elif action['action_type'] == 2: # Use hero power
            if self.current_player.hero.power.requires_target():
                target_info = self.current_player.hero.power.targets[action['target_index']]
                target = f"{action['target_index']} {target_info.name_enUS} (Attack: {target_info.atk} - Health: {target_info.health}/{target_info.max_health})"
            attacker = None
            
        elif action['action_type'] == 3: # Attack with minion
            attacker_info = self.current_player.field[action['attacker_index']]
            attacker = f"{action['attacker_index']} {attacker_info.name_enUS} (Attack: {attacker_info.atk} - Health: {attacker_info.health}/{attacker_info.max_health})"
            target_info = attacker_info.targets[action['target_index']]
            target = f"{action['target_index']} {target_info.name_enUS} (Attack: {target_info.atk} - Health: {target_info.health}/{target_info.max_health})"
            
        elif action['action_type'] == 4: # Attack with hero
            target_info = self.current_player.hero.targets[action['target_index']]
            target = f"{action['target_index']} {target_info.name_enUS} (HAttack: {target_info.atk} - Health: {target_info.health}/{target_info.max_health})"
            attacker = f"{action['attacker_index']} {self.current_player.hero.name_enUS} (Attack: {self.current_player.hero.atk}, Health: {self.current_player.hero.health}, Armor: {self.current_player.hero.armor})"
                
        # Extract the chosen card from a Discover effect
        discover = None
        if self.current_player.choice:
            discover_info = self.current_player.choice.cards[action['discover_index']]
            
            # Check if discover option has a cost
            if discover_info.cost:
                discover = f"{discover_info.name_enUS} ({discover_info.cost} Mana)"
            else:
                discover = f"{discover_info.name_enUS}"
            
            # Try to extract the description of the discover option
            try:
                discover += f" '{discover_info.description.replace('<b>', '').replace('</b>', '')}'"
            except:
                pass
                
            action_type = "Discover"
            
        # Extract the chosen card from a choice effect (Hero Power or card)
        choose = None
        if card_info:
            if card_info.must_choose_one:
                choose_info = card_info.choose_cards[action['choose_index']]
                
                # Check if choose option has a cost
                if choose_info.cost:
                    choose = f"{choose_info.name_enUS} ({choose_info.cost} Mana) '{choose_info.description.replace('<b>', '').replace('</b>', '')}'"
                else:
                    # choose = f"{choose_info.name_enUS}: '{choose_info.description.replace("<b>", "").replace("</b>", "")}'"
                    choose = f"{choose_info.name_enUS}: '{choose_info.description}'"
            
        # Print the action, remove None values
        print_string = f"{index}: {action_type}"
        if action_type == "Use hero power":
            # print_string += f" ({self.current_player.hero.power.name_enUS} '{self.current_player.hero.power.description.replace('<b>', '').replace('</b>', '').replace('\n', ' ')}')"
            print_string += f" ({self.current_player.hero.power.name_enUS} '{self.current_player.hero.power.description}')"
        for name, value in {"Card": card, "Discover Option": discover, "Choose Option": choose, "Attacker": attacker, "Target": target}.items():
            if value:
                print_string += f" - {name}: {value}"
                
        return print_string
    
    def close(self):
        pass
    
    def _get_obs(self) -> np.ndarray:
        """ Build and return the observation vector. """
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        indices = self.obs_indices
        cp = self.current_player  # cache current player
        opponent = cp.opponent
        
        # 1. Basic Player Info
        class_encoding = np.zeros(N_CARD_CLASSES, dtype=np.float32)
        class_encoding[cp.hero.card_class] = 1
        
        observation[indices['player_class_start']:indices['player_class_start'] + N_CARD_CLASSES] = class_encoding
        observation[indices['player_mana']] = cp.mana
        observation[indices['max_mana']] = cp.max_mana
        observation[indices['hero_health']] = cp.hero.health
        observation[indices['hero_attack']] = cp.hero.atk
        observation[indices['hero_armor']] = cp.hero.armor
        observation[indices['has_choice']] = 1 if cp.choice else 0

        # 2. Opponent Hero Health
        observation[indices['opponent_health']] = cp.opponent.hero.health
        observation[indices['opponent_mana']] = opponent.mana
        observation[indices['opponent_max_mana']] = opponent.max_mana
        observation[indices['opponent_hand_size']] = len(opponent.hand)

        # Global game context (e.g., current turn and phase)
        observation[indices['turn_number']] = self.game.turn

        # Hero power availability
        observation[indices['hero_power_available']] = 1 if cp.hero.power.is_usable() else 0

        # Hero weapon details (if applicable)
        if hasattr(cp.hero, 'weapon') and cp.hero.weapon is not None:
            observation[indices['hero_weapon_attack']] = cp.hero.weapon.atk
            observation[indices['hero_weapon_durability']] = cp.hero.weapon.durability
        else:
            observation[indices['hero_weapon_attack']] = 0
            observation[indices['hero_weapon_durability']] = 0

        # 6. Player Board Minions (Stats: attack, health, max_health, can_attack)
        start = indices['board_stats_start']
        for i in range(self.MAX_BOARD_SIZE):
            if i < len(cp.field):
                minion = cp.field[i]
                observation[start + i * N_MINION_STATS + 0] = minion.atk
                observation[start + i * N_MINION_STATS + 1] = minion.health
                observation[start + i * N_MINION_STATS + 2] = minion.max_health
                observation[start + i * N_MINION_STATS + 3] = 1 if minion.can_attack() else 0
                observation[start + i * N_MINION_STATS + 4] = 1 if minion.divine_shield else 0
                observation[start + i * N_MINION_STATS + 5] = 1 if minion.stealthed else 0
                observation[start + i * N_MINION_STATS + 6] = 1 if minion._frozen else 0
                observation[start + i * N_MINION_STATS + 7] = 1 if minion.silenced else 0
                observation[start + i * N_MINION_STATS + 8] = 1 if minion.windfury else 0
                observation[start + i * N_MINION_STATS + 9] = 1 if minion.mega_windfury else 0
                observation[start + i * N_MINION_STATS + 10] = 1 if minion.immune_while_attacking else 0
                observation[start + i * N_MINION_STATS + 11] = 1 if minion.has_inspire else 0
                observation[start + i * N_MINION_STATS + 12] = 1 if minion.has_battlecry else 0
                observation[start + i * N_MINION_STATS + 13] = 1 if minion.has_deathrattle else 0
                observation[start + i * N_MINION_STATS + 14] = 1 if minion.has_overkill else 0
                observation[start + i * N_MINION_STATS + 15] = 1 if minion.lifesteal else 0
            else:
                observation[start + i * N_MINION_STATS:start + i * N_MINION_STATS + N_MINION_STATS] = 0

        # 8. Opponent Board Minions (Stats)
        start = indices['opp_board_stats_start']
        for i in range(self.MAX_BOARD_SIZE):
            if i < len(opponent.field):
                minion = opponent.field[i]
                observation[start + i * N_MINION_STATS + 0] = minion.atk
                observation[start + i * N_MINION_STATS + 1] = minion.health
                observation[start + i * N_MINION_STATS + 2] = minion.max_health
                observation[start + i * N_MINION_STATS + 3] = 1 if minion.can_attack() else 0
                observation[start + i * N_MINION_STATS + 4] = 1 if minion.divine_shield else 0
                observation[start + i * N_MINION_STATS + 5] = 1 if minion.stealthed else 0
                observation[start + i * N_MINION_STATS + 6] = 1 if minion._frozen else 0
                observation[start + i * N_MINION_STATS + 7] = 1 if minion.silenced else 0
                observation[start + i * N_MINION_STATS + 8] = 1 if minion.windfury else 0
                observation[start + i * N_MINION_STATS + 9] = 1 if minion.mega_windfury else 0
                observation[start + i * N_MINION_STATS + 10] = 1 if minion.immune_while_attacking else 0
                observation[start + i * N_MINION_STATS + 11] = 1 if minion.has_inspire else 0
                observation[start + i * N_MINION_STATS + 12] = 1 if minion.has_battlecry else 0
                observation[start + i * N_MINION_STATS + 13] = 1 if minion.has_deathrattle else 0
                observation[start + i * N_MINION_STATS + 14] = 1 if minion.has_overkill else 0
                observation[start + i * N_MINION_STATS + 15] = 1 if minion.lifesteal else 0
            else:
                observation[start + i * N_MINION_STATS:start + i * N_MINION_STATS + N_MINION_STATS] = 0

        # 9. Deck Inclusion (Optional)
        if self.deck_include:
            if cp.name == "Player1":
                player_deck = self.player1_deck
                collection = self.card_collections[0]
            else:
                player_deck = self.player2_deck
                collection = self.card_collections[1]
            start = indices['deck_start']
            deck_indices = np.array([self.card_id_to_index.get(cid, 0) for cid in player_deck])
            counts = np.bincount(deck_indices, minlength=len(collection))
            observation[start:start + len(collection)] = counts[:len(collection)]

        # 9. Deck Inclusion v2 (Optional)
        # Provide ids of the cards in the deck for the current player
        if self.deck_include_v2:
            emb_size = self.embedding_size
            start = indices['deck_start_v2']
            deck_embeddings = np.zeros(emb_size)
            for i in range(self.MAX_DECK_LIMIT):
                if i < len(cp.deck):
                    card_id = cp.deck[i].id
                    data_index = self.card_id_to_index.get(card_id, 0)
                    embedding = self.cards_embedded['embeddings'][data_index]
                    deck_embeddings += embedding
                else:
                    deck_embeddings += 0
            deck_embeddings /= len(cp.deck) if len(cp.deck) > 0 else 1
            observation[start:start + emb_size] = deck_embeddings

        # 10. Embedded Card Features (Optional)
        if self.embedded:
            emb_size = self.embedding_size
            # Hand embeddings (take mean of the hand card embeddings)
            start = indices['embedding_hand_start']
            hand_embeddings = np.zeros(emb_size)
            for i in range(self.MAX_HAND_SIZE):
                if i < len(cp.hand):
                    card_id = cp.hand[i].id
                    data_index = self.card_id_to_index.get(card_id, 0)
                    embedding = self.cards_embedded['embeddings'][data_index]
                    hand_embeddings += embedding
                else:
                    hand_embeddings += 0
                    
            hand_embeddings /= len(cp.hand) if len(cp.hand) > 0 else 1
            observation[start:start + emb_size] = hand_embeddings

            # Player Board embeddings
            start = indices['embedding_board_start']
            board_embeddings = np.zeros(emb_size)
            for i in range(self.MAX_BOARD_SIZE):
                if i < len(cp.field):
                    minion_id = cp.field[i].id
                    data_index = self.card_id_to_index.get(minion_id, 0)
                    embedding = self.cards_embedded['embeddings'][data_index]
                    board_embeddings += embedding
                else:
                    board_embeddings += 0
            board_embeddings /= len(cp.field) if len(cp.field) > 0 else 1
            observation[start:start + emb_size] = board_embeddings

            # Opponent Board embeddings
            start = indices['embedding_opp_board_start']
            opp_board_embeddings = np.zeros(emb_size)
            for i in range(self.MAX_BOARD_SIZE):
                if i < len(opponent.field):
                    minion_id = opponent.field[i].id
                    data_index = self.card_id_to_index.get(minion_id, 0)
                    embedding = self.cards_embedded['embeddings'][data_index]
                    opp_board_embeddings += embedding
                else:
                    opp_board_embeddings += 0
                    
            opp_board_embeddings /= len(opponent.field) if len(opponent.field) > 0 else 1
            observation[start:start + emb_size] = opp_board_embeddings
            
            # # Take mean of the hand and board embeddings to create a single embedding vector
            # final_embedding = hand_embeddings + board_embeddings + opp_board_embeddings
            # # Divide by 3 to normalize
            # final_embedding /= 3
            # observation[start:start + emb_size] = final_embedding       

        return observation

    
    def _get_info(self):
        pass
    
    def _calculate_final_reward(self) -> float:
        """
        Internal function to calculate the reward based on the current state of the environment
        
        :return: The reward based on the current state of the environment
        """
        
        if self.final_reward_mode == 0:
            bonus = 10.0
            penalty = -10.0
        elif self.final_reward_mode == 1:
            bonus = 100.0
            penalty = -100.0
        else:
            bonus = 1.0
            penalty = -1.0
        
        # If the game has ended, return the reward based on the player's playstate (WON, LOST, TIED)
        if self.game.ended:
            if self.current_player.playstate == PlayState.WON:
                return bonus, "won"
            else:
                return penalty, "lost"
            
        # Double check if the game has ended by checking player health
        if self.current_player.hero.health <= 0 and self.current_player.opponent.hero.health <= 0:
            return 0.0, "tied"
        elif self.current_player.hero.health <= 0:
            return penalty, "lost"
        elif self.current_player.opponent.hero.health <= 0:
            return bonus, "won"            
        else:
            return 0.0, "not_done"
        
    def _update_previous_state_metrics(self):
        """ Update the stored previous state metrics for the next step. """
        self.prev_hero_healths = [player.hero.health for player in self.game.players]
        self.prev_minions = [len(player.field) for player in self.game.players]
        self.prev_mana = [player.mana for player in self.game.players]
        self.prev_hand_sizes = [len(player.hand) for player in self.game.players]
        
    

    def _calculate_intermediate_reward(self) -> float:
        """
        Internal function to calculate the intermediate reward based on the current state of the environment
        :return: The intermediate reward based on the current state of the environment
        """
        if self.incremental_reward_mode == 0:
            return 0.0
        
        # Build current state arrays
        current_hero_healths = np.array([player.hero.health for player in self.game.players])
        current_minions = np.array([len(player.field) for player in self.game.players])
        current_mana = np.array([player.mana for player in self.game.players])
        current_hand_size = len(self.current_player.hand)
        reward = calculate_intermediate_reward_numba(
            self.incremental_reward_mode,
            np.array(self.prev_hero_healths), current_hero_healths,
            np.array(self.prev_minions), current_minions,
            np.array(self.prev_mana), current_mana,
            np.array(self.prev_hand_sizes)[0], current_hand_size
        )
        
        return reward
        
    def get_valid_actions(self) -> tuple:
        """
        Get the valid actions for the current player
        
        :return: The valid actions and the action mask
        """
        
        valid_actions = []
        
        # If the player has a choice, add all possible choices
        # No other actions are possible, so return the valid actions and action mask
        if self.current_player.choice:
            for i in range(len(self.current_player.choice.cards)):
                valid_actions.append({
                    'action_type': 0,
                    'card_index': 0,
                    'attacker_index': 0,
                    'target_index': 0,
                    'discover_index': i,
                    'choose_index': 0
                })
                
            action_mask = self._get_action_mask(valid_actions)
                
            return valid_actions, action_mask
        
        # End turn
        valid_actions.append({
            'action_type': 0,
            'card_index': 0,
            'attacker_index': 0,
            'target_index': 0,
            'discover_index': 0,
            'choose_index': 0
        })
        
        # Play card
        for i, card in enumerate(self.current_player.hand):
            if card.is_playable():
                # If the card requires a target, add all possible targets
                # If the card requires a choice, add all possible choices
                # Otherwise, add the action with default values
                if card.must_choose_one:
                    for j, choice in enumerate(card.choose_cards):
                        if choice.requires_target():
                            for k, target in enumerate(choice.targets):
                                valid_actions.append({
                                    'action_type': 1,
                                    'card_index': i,
                                    'attacker_index': 0,
                                    'target_index': k,
                                    'discover_index': 0,
                                    'choose_index': j
                                })
                        else:
                            valid_actions.append({
                                'action_type': 1,
                                'card_index': i,
                                'attacker_index': 0,
                                'target_index': 0,
                                'discover_index': 0,
                                'choose_index': j
                            })
                            
                elif card.requires_target():
                    for j, target in enumerate(card.targets):
                        valid_actions.append({
                            'action_type': 1,
                            'card_index': i,
                            'attacker_index': 0,
                            'target_index': j,
                            'discover_index': 0,
                            'choose_index': 0
                        })
                else:
                    valid_actions.append({
                        'action_type': 1,
                        'card_index': i,
                        'attacker_index': 0,
                        'target_index': 0,
                        'discover_index': 0,
                        'choose_index': 0
                    })
                    
        # Use hero power
        heropower = self.current_player.hero.power
        if heropower.is_usable():
            if heropower.requires_target():
                for i, target in enumerate(heropower.targets):
                    valid_actions.append({
                        'action_type': 2,
                        'card_index': 0,
                        'attacker_index': 0,
                        'target_index': i,
                        'discover_index': 0,
                        'choose_index': 0
                    })
            elif heropower.must_choose_one:
                for i, choice in enumerate(heropower.choose_cards):
                    valid_actions.append({
                        'action_type': 2,
                        'card_index': 0,
                        'attacker_index': 0,
                        'target_index': 0,
                        'discover_index': 0,
                        'choose_index': i
                    })
            else:
                valid_actions.append({
                    'action_type': 2,
                    'card_index': 0,
                    'attacker_index': 0,
                    'target_index': 0,
                    'discover_index': 0,
                    'choose_index': 0
                })
                
        # Attack with minion
        for i, minion in enumerate(self.current_player.field):
            if minion.can_attack():
                for j, target in enumerate(minion.targets):
                    valid_actions.append({
                        'action_type': 3,
                        'card_index': 0,
                        'attacker_index': i,
                        'target_index': j,
                        'discover_index': 0,
                        'choose_index': 0
                    })
                    
        # Attack with hero
        hero = self.current_player.hero
        if hero.can_attack():
            for i, target in enumerate(hero.targets):
                valid_actions.append({
                    'action_type': 4,
                    'card_index': 0,
                    'attacker_index': 0,
                    'target_index': i,
                    'discover_index': 0,
                    'choose_index': 0
                })
                
        action_mask = self._get_action_mask(valid_actions)
                
        return valid_actions, action_mask
    
    def _get_action_mask(self, valid_actions: list) -> np.ndarray:
        """
        Internal function to get the action mask for the valid actions
        The list is converted to a binary mask where 1 indicates a valid action and 0 indicates an invalid action
        
        :param valid_actions: The valid actions
        :return: The action mask
        """
                
        # Initialize the action mask with zeros
        action_mask = np.zeros(self.num_flat_actions, dtype=np.int64)
        
        # Set the action mask to 1 for valid actions
        for action in valid_actions:
            idx = self._action_to_index(action)
            action_mask[idx] = 1
        
        return action_mask
    
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

    def flat_index_to_action_dict(self, idx: int) -> dict:
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
        
    def _resolve_choices(self, discover_index: int):
        """ Resolve outstanding choices before or after the action. """
        cp = self.current_player
        while cp.choice:
            # Resolve the choice in one call if possible
            cp.choice.choose(cp.choice.cards[discover_index])
            
    def _clean_board(self):
        """ Clean the board of any minions that have died. """
        self.game.process_deaths()
            
    