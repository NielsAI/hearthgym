"""
This script runs a Hearthstone game simulation with specified settings and agents provided as command line arguments.

The script uses the Hearthstone environment and agents from the `env` and `agents` modules, respectively. The agents are defined in the `agents` module, and the environment is defined in the `env` module.

Agemts:
- 0 RandomAgent: Random agent that selects actions uniformly at random.
- 1 HumanAgent: Human agent that allows manual input for selecting actions.
- 2 DynamicLookaheadAgent: Agent that uses dynamic lookahead to evaluate the best action.
- 3 GretiveCompAgent: Agent that uses a greedy choice approach to select actions.
- 4 WeightedScoreAgent: Agent that selects actions based on a weighted score.
- 5 NaiveScoreLookaheadAgent: Agent that uses a naive score lookahead approach.
- 6 PPOAgent: Masked Proximal Policy Optimization (PPO) agent for reinforcement learning.
- 7 GreedyAgent: Greedy agent that selects the best action based on a scoring method.
- 8 BaseDynamicLookaheadAgent: Base class for dynamic lookahead agents.
- 9 WorldModelAgent: Agent that uses a world model for simulation and decision-making.
- 10 EncodedPPOAgent: Encoded version of the PPO agent for reinforcement learning.

Classes:
- 1 Deathknight (DISABLED)
- 2 Druid (0: Token/Swarm | 1: Jade Golem)
- 3 Hunter (0: Midrange Beast | 1: Deathrattle)
- 4 Mage (0: Tempo/Elemental | 1: Big Spell/Controll)
- 5 Paladin (0: Silver Hand Recruit | 1: Control/N'Zoth)
- 6 Priest (0: Dragon | 1: Combo/Resurrect)
- 7 Rogue (0: Tempo/Weapon | 1: Miracle/Gadgetzan)
- 8 Shaman (0: Totem & Overload | 1: Control/Big Spells)
- 9 Warlock (0: Zoo/Discard | 1: Control/Demon)
- 10 Warrior (0: Tempo/Taunt | 1: Mech/Control)

The script takes the following command line arguments:

- `--agent1` (`-a1`): Index of the first agent class (default: 0)
- `--agent2` (`-a2`): Index of the second agent class (default: 0)
- `--class1` (`-c1`): Class of the first player (default: 2)
- `--class2` (`-c2`): Class of the second player (default: 2)
- `--games` (`-g`): Number of games to play (default: 1)
- `--log_file` (`-l`): Path to log file (default: 'logs/game.log')
- `--render` (`-r`): Render the game environment (default: False)
- `--intern_logging` (`-i`): Use internal logging (default: False)
- `--seed` (`-s`): Random seed for reproducibility (default: None)
- `--embed` (`-e`): Use embedded mode (default: False)
- `--deck_include` (`-d`): Include deck information in the environment (default: False)
- `--deck1` (`-d1`): Deck ID for the first player (default: 0)
- `--deck2` (`-d2`): Deck ID for the second player (default: 0)
- `--ppo_model1` (`-p1`): PPO model for the first player (default: None)
- `--ppo_model2` (`-p2`): PPO model for the second player (default: None)
- `--ppo_type1` (`-pt1`): PPO type for the first player (default: "Mask")
- `--ppo_type2` (`-pt2`): PPO type for the second player (default: "Mask")
- `--score_method1` (`-s1`): Score method for the first player (default: "aggro")
- `--score_method2` (`-s2`): Score method for the second player (default: "aggro")
- `--help`: Show help message and exit


The script initializes the Hearthstone environment and agents, sets up logging, and runs the game simulation. It also handles dynamic class selection, deck selection, and agent instantiation based on the provided command line arguments.
The script then runs the game simulation with the specified settings and agents.
"""

# Import Fireplace modules
from fireplace.logging import log
from fireplace import cards

# Import the Hearthstone agents
from agents.RandomAgent import RandomAgent # RandomAgent implementation
from agents.HumanAgent import HumanAgent # HumanAgent implementation
from agents.DLAgents import DynamicLookaheadAgent # Dynamic Lookahead Agent implementation
from agents.GCAgents import GretiveCompAgent # Greedy Choice Agent implementation
from agents.SMAgents import NaiveScoreLookaheadAgent, WeightedScoreAgent # Sebastian Miller's agents
from agents.PPOAgent import PPOAgent # Proximal Policy Optimization (PPO) Agent implementation
from agents.GreedyAgent import GreedyAgent # Greedy Agent implementation
from agents.BaseDLAgent import BaseDynamicLookaheadAgent # Base Dynamic Lookahead Agent implementation
from agents.WorldModelAgent import WorldModelAgent # World Model Agent implementation
from agents.EncodedPPOAgent import EncodedPPOAgent # Encoded PPO Agent implementation

# Import required helper functions
from hearthstone.enums import PlayState
from functions.logging import setup_logging, log_event
from functions.data_loading import generate_feature_names

# Import general libraries
from tqdm import tqdm
import pandas as pd
import logging
import argparse
import random

# Import gymnasium for creating the Hearthstone environment
import gymnasium

# This is to avoid the Simplex error when using the PPO agents
from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)

# Setup gymnasium
from gymnasium.envs.registration import register

register(
    id="hearthstone_env/HearthGym-v0",
    entry_point="env.hearthstone.HearthGym:HearthstoneEnv"
)

def run_game(
    agent_options: dict,
    agent1_index: int = 0,
    agent2_index: int = 0,
    games: int = 1, 
    seed: int = None, 
    log_file: str = "logs/game.log", 
    obs_file: str = None,
    data_file: str = "src/data/card_data.csv",
    render: bool = False,
    class_options: dict = None,
    class1: int = None, 
    class2: int = None,
    ppo_model1: str = None,
    ppo_model2: str = None,
    ppo_type1: str = "Mask",
    ppo_type2: str = "Mask",
    embedded: bool = False,
    deck_include: bool = False,
    deck_include_v2: bool = False,
    deck1: str = "0",
    deck2: str = "0",
    score_method1: str = "aggro",
    score_method2: str = "aggro",
    mirror: bool = False,
    encoder1: str = None,
    encoder2: str = None,
    rssm1: str = None,
    rssm2: str = None,
):
    """
    Run a Hearthstone game simulation with the specified agents.
    Optimizations include caching of CSV data, deck metadata lookups, and
    reduction of redundant operations in the game loop.
    """
    setup_logging(log_file)
    if obs_file is not None:
        setup_logging(obs_file)
    
        # Include list with flags for environment modes in observation
        mode_flags = {
            "embedded": embedded,
            "deck_include": deck_include,
            "deck_include_v2": deck_include_v2
            }
        
        log_event(event=mode_flags, log_file=obs_file)
    
    wins = {0: 0, 1: 0}
    
    if seed is not None:
        random.seed(seed)
    
    print("Initializing Hearthstone environment...")
    print("-" * 75)
    
    # Cache CSV data (read once)
    card_data = pd.read_csv(data_file)
    decks = pd.read_csv('src/data/final_decks.csv')
    deck_metadata = pd.read_csv('src/data/decks_metadata.csv')
    
    # Precompute available decks by hero class
    available_decks = {}
    for cls in class_options.values():
        available_decks[cls] = deck_metadata[deck_metadata['hero_class'] == cls]

    _, cont_indices, disc_indices = generate_feature_names(
            embedded        = embedded,
            deck_include    = deck_include,
            deck_include_v2 = deck_include_v2,
            embedding_size  = 368)

    print("Creating agents...")
    print("-" * 75)
    
    # Create agent instances (once)
    agent1_class_name = list(agent_options.keys())[agent1_index]
    if agent1_class_name == "PPOAgent" and ppo_model1 is not None:
        agent1 = agent_options[agent1_class_name](ppo_model1, ppo_type1)
    elif agent1_class_name == "GreedyAgent":
        agent1 = agent_options[agent1_class_name](score_method1)
    elif agent1_class_name == "WorldModelAgent":
        agent1 = agent_options[agent1_class_name](
            cont_indices=cont_indices,
            disc_indices=disc_indices,
            encoder_path=encoder1,
            rssm_path=rssm1,
            controller_path=ppo_model1,
            ppo_model_type=ppo_type1
        )
    elif agent1_class_name == "EncodedPPOAgent":
        agent1 = agent_options[agent1_class_name](
            cont_indices=cont_indices,
            disc_indices=disc_indices,
            encoder_path=encoder1,
            model_path=ppo_model1,
            model_type=ppo_type1
        )
    else:
        agent1 = agent_options[agent1_class_name]()
    
    agent2_class_name = list(agent_options.keys())[agent2_index]
    if agent2_class_name == "PPOAgent" and ppo_model2 is not None:
        agent2 = agent_options[agent2_class_name](ppo_model2, ppo_type2)
    elif agent2_class_name == "GreedyAgent":
        agent2 = agent_options[agent2_class_name](score_method2)
    elif agent2_class_name == "WorldModelAgent":
        agent2 = agent_options[agent2_class_name](
            encoder_path=encoder2,
            rssm_path=rssm2,
            controller_path=ppo_model2,
            ppo_model_type=ppo_type2
        )
    elif agent2_class_name == "EncodedPPOAgent":
        agent2 = agent_options[agent2_class_name](
            encoder_path=encoder2,
            model_path=ppo_model2,
            model_type=ppo_type2
        )
    else:
        agent2 = agent_options[agent2_class_name]()
    
    # If a human agent is used, force rendering
    if agent1_class_name == "HumanAgent" or agent2_class_name == "HumanAgent":
        render = True
    
    # Convert deck arguments once
    if deck1 != "all":
        deck1 = int(deck1)
    
    if deck2 != "all":
        deck2 = int(deck2)
    
    # Setup player info
    card_classes = [None, None]
    if class1 != "all":
        selected_class1 = int(class1)
        card_classes[0] = class_options[selected_class1]
        
        available_decks1 = available_decks[card_classes[0]]
        
        print_class1 = class_options[selected_class1]    
        print_deck1 = available_decks1["deck_name"].values[deck1]    
    else:
        print_class1 = "Random"
        
        if deck1 == "all":
            print_deck1 = "Random"
        else:
            print_deck1 = "First Deck"
    
    if mirror:
        print("Mirror match detected. Using the same class and deck for both players.")
        print(f"Player 1 and Player 2 will use the same class and deck: {print_class1} - {print_deck1}")
        class2 = class1
        deck2 = deck1
    
    if class2 != "all":
        selected_class2 = int(class2)
        card_classes[1] = class_options[selected_class2]
        
        available_decks2 = available_decks[card_classes[1]]
        
        print_class2 = class_options[selected_class2]
        print_deck2 = available_decks2["deck_name"].values[deck2]
    else:
        print_class2 = "Random"
        
        if deck2 == "all":
            print_deck2 = "Random"
        else:
            print_deck2 = "First Deck"
    
    # Store the agents and their classes
    agents = [agent1, agent2]
    agent_classes = [agent1_class_name, agent2_class_name]
    if agent1_class_name == "GreedyAgent":
        agent_classes[0] = f"GreedyAgent ({score_method1})"
    if agent2_class_name == "GreedyAgent":
        agent_classes[1] = f"GreedyAgent ({score_method2})"
    
    player_info = {
        i: {
            "name": f"Player {i+1}",
            "agent_class": agent_classes[i],
            "card_class": card_classes[i]
        } for i in range(2)
    }
    
    print(f"Player 1 Agent: {player_info[0]["agent_class"]} --- Class: {print_class1} --- Deck: {print_deck1}")
    print(f"Player 2 Agent: {player_info[1]["agent_class"]} --- Class: {print_class2} --- Deck: {print_deck2}")
    
    print("-" * 75)              
    print("Starting game simulation...")
    
    # Initialize progress indicator
    game_iter = tqdm(range(1, games + 1), desc="Games", unit="game") if not render else range(1, games + 1)
    
    for game_number in game_iter:
        # Dynamic class selection if "all" is chosen
        if class1 == "all":
            selected_class1 = random.choice(list(class_options.keys()))
            card_classes[0] = class_options[selected_class1]
            player_info[0]["card_class"] = class_options[selected_class1]
            
        if mirror:
            selected_class2 = selected_class1
            card_classes[1] = card_classes[0]
            player_info[1]["card_class"] = player_info[0]["card_class"]
        else:
            if class2 == "all":
                selected_class2 = random.choice(list(class_options.keys()))
                card_classes[1] = class_options[selected_class2]
                player_info[1]["card_class"] = class_options[selected_class2]
        
        # Get deck lists based on precomputed available_decks
        if deck1 == "all":
            available = available_decks[card_classes[0]]
            selected_deck1 = random.choice(range(len(available)))
        else:
            selected_deck1 = deck1
        deck_id1 = available_decks[card_classes[0]]["deck_id"].values[selected_deck1]
        deck1_list = decks[decks['deck_id'] == deck_id1]['card_id'].values
        print_deck1 = available_decks[card_classes[0]]["deck_name"].values[selected_deck1]
    
        if mirror:
            selected_deck2 = selected_deck1
            deck_id2 = deck_id1
            deck2_list = deck1_list
            print_deck2 = print_deck1
        else:
            if deck2 == "all":
                available = available_decks[card_classes[1]]
                selected_deck2 = random.choice(range(len(available)))
            else:
                selected_deck2 = deck2
            deck_id2 = available_decks[card_classes[1]]["deck_id"].values[selected_deck2]
            deck2_list = decks[decks['deck_id'] == deck_id2]['card_id'].values
            print_deck2 = available_decks[card_classes[1]]["deck_name"].values[selected_deck2]
    
        # Initialize environment once per game (passes pre-read card_data)
        env = gymnasium.make(
            id                      = "hearthstone_env/HearthGym-v0",
            class1                  = selected_class1,
            class2                  = selected_class2,
            clone                   = None,
            card_data               = card_data,
            final_reward_mode       = 2,
            incremental_reward_mode = 0,
            embedded                = embedded,
            deck_include            = deck_include,
            deck_include_v2         = deck_include_v2,
            deck1                   = deck1_list,
            deck2                   = deck2_list,
            class_options           = class_options,
        ).unwrapped
    
        observation, info = env.reset()
    
        # Preload the action space for agents
        for agent in agents:
            agent.load_action_space(env.action_space)
            if hasattr(agent, "new_game"):
                agent.new_game(env)
        
        done = False
        env.current_player = env.game.current_player
        turn = 0
        action = 0
        
        while not done:
            turn += 1
            current_player_index = env.game.players.index(env.current_player)
            player_name = player_info[current_player_index]["name"]
            agent = agents[current_player_index]
            
            if not render:
                # Use current_player index from start to avoid unnecessary lookups
                player_name = player_info[current_player_index]["name"]
                
                # Get the longest agent class name length used for formatting
                max_agent_class_length = max([len(agent_class) for agent_class in agent_classes])
                
                game_iter.set_description(
                    f"Turn {turn:3d} | {player_name:<10} - {agent_classes[current_player_index]:<{max_agent_class_length}} | "
                    f"P1 HP: {env.game.players[0].hero.health:3d} - P2 HP: {env.game.players[1].hero.health:3d}"
                )
            
            if render:
                print(f"\n{player_name}'s turn:")
                env.render()
            
            valid_actions, action_mask = env.get_valid_actions()
            # If the agent is a WorldModelAgent, add the previous action
            action = agent.act(
                observation=observation, 
                valid_actions=valid_actions, 
                action_mask=action_mask, 
                env=env
                )
                
            log_action = env.flat_index_to_action_dict(action.copy()) if not isinstance(action, dict) else action
    
            if render:
                try:
                    print(f"Action chosen:")
                    print(log_action)
                    env.render_actions([log_action])
                except Exception as e:
                    print(f"Error rendering action: {e}")
    
            try:
                action_string = env._action_to_string(index=0, action=log_action)
            except Exception as e:
                print(f"Error converting action to string: {e}")
                action_string = str(log_action)
            
            observation, reward, done, truncated, info = env.step(action=action)
    
            event = {
                'turn': turn,
                'game_number': game_number,
                'player': player_name,
                'card_class': player_info[current_player_index]["card_class"],
                'agent_type': agent_classes[current_player_index],
                'action_valid': info.get("valid_action", True),
                'action': log_action,
                'action_string': action_string,
                'reward': reward,
                'done': done,
                'player_result': info.get("player_result", "not_done"),
                'deck': [print_deck1, print_deck2][current_player_index]
            }
    
            if agent_classes[current_player_index].startswith("PPOAgent"):
                event["ppo_model"] = (ppo_model1 if current_player_index == 0 else ppo_model2)
            else:
                event["ppo_model"] = None
    
            if agent_classes[current_player_index].startswith("GreedyAgent"):
                event["score_method"] = (score_method1 if current_player_index == 0 else score_method2)
            else:
                event["score_method"] = None
    
            # Log the event and observation
            log_event(event=event, log_file=log_file)
            
            if obs_file is not None:
                # Convert ndarray observation to list for logging
                log_event(event=observation.tolist(), log_file=obs_file)
    
            if done:
                break
            
        # Determine the winner
        if render:
            print("\nGame Finished.")
        for i, player in enumerate(env.game.players):
            if player.playstate == PlayState.WON:
                winner_name = player_info[i]["name"]
                winner_class = player_info[i]["agent_class"]
                winner_card_class = player_info[i]["card_class"]
                if render:
                    print(f"{winner_name} ({winner_class} - {winner_card_class}) wins!")
                wins[i] += 1
                break
        else:
            if render:
                print("The game ended in a draw.")
    
     # Display final results
    print("\nFinal Results:")
    print_classes = card_classes.copy()
    
    if ppo_model1 is not None:
        print("PPO Model 1:", ppo_model1)
    if ppo_model2 is not None:
        print("PPO Model 2:", ppo_model2)
    
    if class1 == "all":
        print_classes[0] = "Random"
    if class2 == "all":
        print_classes[1] = "Random"
    
    # Display the final results for each agent
    for agent_index, win_count in wins.items():
        agent_name = player_info[agent_index]["name"]
        agent_class = player_info[agent_index]["agent_class"]
        agent_card_class = print_classes[agent_index]
        print(f"{agent_name} ({agent_class} - {agent_card_class}): {win_count} wins")
        
    print("\nGame simulation complete. The log file has been saved to:", log_file)
    if obs_file is not None:
        print("Observations have been saved to:", obs_file)
    
if __name__ == "__main__":
    # Define the agents to be used in the simulation
    agent_options = {
        "RandomAgent": RandomAgent,
        "HumanAgent": HumanAgent,
        "DynamicLookaheadAgent": DynamicLookaheadAgent,
        "GretiveCompAgent": GretiveCompAgent,
        "WeightedScoreAgent": WeightedScoreAgent,
        "NaiveScoreLookaheadAgent": NaiveScoreLookaheadAgent,
        "PPOAgent": PPOAgent,
        "GreedyAgent": GreedyAgent,
        "BaseDynamicLookaheadAgent": BaseDynamicLookaheadAgent,
        "WorldModelAgent": WorldModelAgent,
        "EncodedPPOAgent": EncodedPPOAgent,
    }
    
    # Create display options for the agent selection in the help message ([index] - [agent_name])
    display_agents = [f"{i} - {agent_name}" for i, agent_name in enumerate(agent_options.keys())]
    
    # Classes:
    # 1: Deathknight (DISABLED), 2: Druid, 3: Hunter, 4: Mage, 5: Paladin, 
    # 6: Priest, 7: Rogue, 8: Shaman, 9: Warlock, 10: Warrior 11: Dream (DISABLED), 
    # 12: Neutral (DISABLED) 13: Whizbang (DISABLED), 14: Demonhunter (DISABLED)
    
    # Class Card Collection sizes (total cards in each Cardclass available):
    # 2: 1014, 3: 1015, 4: 1015, 5: 1015, 6: 1015, 7: 1016, 8: 1013, 9: 1016, 10: 1013
    
    # Define the default agent classes
    class_options = {
        2: "Druid",
        3: "Hunter",
        4: "Mage",
        5: "Paladin",
        6: "Priest",
        7: "Rogue",
        8: "Shaman",
        9: "Warlock",
        10: "Warrior"
    }
    
    # Create display options for the class selection in the help message ([index] - [class_name])
    display_classes = [f"{i} - {class_name}" for i, class_name in class_options.items()]
    # Add the "all" option to the class options
    display_classes.append("all - Random Class")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a Hearthstone game simulation.")
    
    # Agents
    parser.add_argument(
        '-a1', '--agent1', type=int, default=0,
        help=f"Index of the first agent (default: 0 ({list(agent_options.keys())[0]})) Options: {display_agents}"
    )
    parser.add_argument(
        '-a2', '--agent2', type=int, default=0,
        help=f"Index of the second agent (default: 0 ({list(agent_options.keys())[0]})) Options: {display_agents}"
    )
    
    # Classes
    parser.add_argument(
        '-c1', '--class1', type=str, default=2,
        help=f"Class of the first player (default: 2 ({class_options[2]})) Options: {display_classes}"
    )
    parser.add_argument(
        '-c2', '--class2', type=str, default=2,
        help=f"Class of the second player (default: 2 ({class_options[2]})) Options: {display_classes}"
    )
    
    # Number of games
    parser.add_argument(
        '-g', '--games', type=int, default=1,
        help="Number of games to play (default: 1)"
    )
    # Log file path
    parser.add_argument(
        '-l', '--log_file', type=str, default='logs/game.log',
        help="Path to log file (default: logs/game.log)"
    )
    # Observation file path
    parser.add_argument(
        '-o', '--obs_file', type=str, default=None,
        help="Path to observation file (default: None)"
    )
    # Game rendering
    parser.add_argument(
        '-r', '--render', type=bool, default=False,
        help="Render the game environment (default: False)"
    )
    # Fireplace logging
    parser.add_argument(
        '-i', '--intern_logging', type=bool, default=False,
        help="Use internal logging (default: False)"
    )
    # Seed
    parser.add_argument(
        '-s', '--seed', type=int, required=False, default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    # OPTIONAL FOR MIRROR MATCHES
    parser.add_argument(
        '-m', '--mirror', type=bool, required=False, default=False,
        help="Use mirror matches (default: False)"
    )
    
    # OPTIONAL FOR EMBEDDED MODE
    parser.add_argument(
        '-e', '--embed', type=bool, required=False, default=False,
        help="Use embedded mode (default: False)"
    )
    
    # OPTIONAL FOR DECK INCLUSION MODE
    parser.add_argument(
        '-d', '--deck_include', type=bool, required=False, default=False,
        help="Include deck information in the environment (default: False)"
    )
    parser.add_argument(
        '-dv2', '--deck_include_v2', type=bool, required=False, default=False,
        help="Include deck information (including order) in the new state space (default: False)"
    )
    
    # OPTIONAL FOR DECK SELECTION
    parser.add_argument(
        '-d1', '--deck1', type=str, required=False, default=0,
        help="Deck ID for the first player (default: 0). Use 'all' for random deck selection."
    )
    parser.add_argument(
        '-d2', '--deck2', type=str, required=False, default=0,
        help="Deck ID for the second player (default: 0). Use 'all' for random deck selection."
    )
    
    # OPTIONAL FOR PPO AGENT
    parser.add_argument(
        '-p1', '--ppo_model1', type=str, required=False, default=None,
        help="PPO model for the first player (default: None)."
    )
    parser.add_argument(
        '-p2', '--ppo_model2', type=str, required=False, default=None,
        help="PPO model for the second player (default: None)."
    )
    parser.add_argument(
        '-pt1', '--ppo_type1', type=str, required=False, default="Mask",
        help="PPO type for the first player (default: Mask; options: Mask, MaskRNN, RNN)."
    )
    parser.add_argument(
        '-pt2', '--ppo_type2', type=str, required=False, default="Mask",
        help="PPO type for the second player (default: Mask; options: Mask, MaskRNN, RNN)."
    )
    
    # OPTIONAL FOR GREEDY BASELINE AGENT
    parser.add_argument(
        '-s1', '--score_method1', type=str, required=False, default="aggro",
        help="Score method for the first player (default: aggro; options: aggro, control, ramp)."
    )
    parser.add_argument(
        '-s2', '--score_method2', type=str, required=False, default="aggro",
        help="Score method for the second player (default: aggro; options: aggro, control, ramp)."
    )
    
    # OPTIONAL FOR WORLD MODEL AGENT and ENCODED PPO AGENT
    parser.add_argument(
        "-we1", "--world_model_encoder1", type=str, required=False, default=None,
        help="Path to the encoder model for the first player (default: None)."
    )
    parser.add_argument(
        "-we2", "--world_model_encoder2", type=str, required=False, default=None,
        help="Path to the encoder model for the second player (default: None)."
    )
    parser.add_argument(
        "-wr1", "--world_model_rssm1", type=str, required=False, default=None,
        help="Path to the RSSM model for the first player (default: None)."
    )
    parser.add_argument(
        "-wr2", "--world_model_rssm2", type=str, required=False, default=None,
        help="Path to the RSSM model for the second player (default: None)."
    )
        
    args = parser.parse_args()
    
    # Disable Fireplace logging if not in debug mode
    if not args.intern_logging:
        log.setLevel(logging.ERROR)
    
    print("Initializing cards...")
    cards.db.initialize()
    
    run_game(
        agent_options   = agent_options,
        agent1_index    = args.agent1, 
        agent2_index    = args.agent2, 
        games           = args.games, 
        seed            = args.seed, 
        log_file        = args.log_file, 
        obs_file        = args.obs_file,
        render          = args.render,
        class_options   = class_options,
        class1          = args.class1,
        class2          = args.class2,
        ppo_model1      = args.ppo_model1,
        ppo_model2      = args.ppo_model2,
        ppo_type1       = args.ppo_type1,
        ppo_type2       = args.ppo_type2,
        embedded        = args.embed,
        deck_include    = args.deck_include,
        deck_include_v2 = args.deck_include_v2,
        deck1           = args.deck1,
        deck2           = args.deck2,
        score_method1   = args.score_method1,
        score_method2   = args.score_method2,
        mirror          = args.mirror,
        encoder1        = args.world_model_encoder1,
        encoder2        = args.world_model_encoder2,
        rssm1           = args.world_model_rssm1,
        rssm2           = args.world_model_rssm2,
        )
