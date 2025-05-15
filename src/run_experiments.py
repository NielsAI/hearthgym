# Load configuration
import yaml
from yaml.loader import SafeLoader

from fireplace.logging import log
import logging

with open('src/experiments/experiments_config.yaml') as config_file:
    experiments_config = yaml.load(config_file, Loader=SafeLoader)

# Import the function for the simulation
from run_game import run_game

# Import the Hearthstone agents
from agents.RandomAgent import RandomAgent # RandomAgent implementation
from agents.HumanAgent import HumanAgent # HumanAgent implementation
from agents.DLAgents import DynamicLookaheadAgent # Dynamic Lookahead Agent implementation
from agents.GCAgents import GretiveCompAgent # Greedy Choice Agent implementation
from agents.SMAgents import NaiveScoreLookaheadAgent, WeightedScoreAgent # Sebastian Miller's agents
from agents.PPOAgent import PPOAgent # Proximal Policy Optimization (PPO) Agent implementation
from agents.GreedyAgent import GreedyAgent # GreedyAgent implementation
from agents.BaseDLAgent import BaseDynamicLookaheadAgent # BaseDynamicLookaheadAgent implementation
from agents.WorldModelAgent import WorldModelAgent as WorldModelAgent # WorldModelAgent implementation
from agents.EncodedPPOAgent import EncodedPPOAgent # Encoded PPO Agent implementation

from fireplace import cards

def main():
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

    # Disable internal logging
    log.setLevel(logging.ERROR)

    # Loop over experiments in the config file and run them
    for i, (experiment_id, config) in enumerate(experiments_config["experiments"].items()):
        if not config["enabled"]:
            continue
        
        # Get the log folder
        log_folder = config["folder"]
        
        print(f"Running experiment: {experiment_id}...")
        run_game(
            agent_options=  agent_options,
            agent1_index=   config['agent1_index'],
            agent2_index=   config['agent2_index'],
            games=          config['games'],
            seed=           config['seed'],
            log_file=       f"logs/{log_folder}/games_{experiment_id}.log", 
            obs_file=       f"logs/{log_folder}/observations_{experiment_id}.log" if config['save_observations'] else None,
            data_file=      "src/data/card_data.csv",
            render=         False,
            class_options=  class_options,
            class1=         config['class1'],
            class2=         config['class2'],
            ppo_model1=     config['ppo_model1'],
            ppo_model2=     config['ppo_model2'],
            ppo_type1=      config['ppo_type1'],
            ppo_type2=      config['ppo_type2'],
            embedded=       config['embedded'],
            deck_include=   config['deck_include'],
            deck_include_v2= config['deck_include_v2'],
            deck1=          config['deck1'],
            deck2=          config['deck2'],
            score_method1=  config['score_method1'],
            score_method2=  config['score_method2'],
            mirror=         config['mirror'],
            encoder1=       config['encoder1'],
            encoder2=       config['encoder2'],
            rssm1=          config['rssm1'],
            rssm2=          config['rssm2'],
            )

if __name__ == "__main__":
    log.setLevel(logging.ERROR)
    
    cards.db.initialize()
    
    main()