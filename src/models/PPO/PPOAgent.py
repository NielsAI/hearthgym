# General libraries
import pandas as pd
import numpy as np
import torch
import json

# Import the RecurrentMaskablePPO model
from models.PPO.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO

# Stable Baselines
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

# Maskable PPO and other sb3_contrib imports
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker

# Custom environment
from env.hearthstone.HearthVsAgentEnv import HearthVsAgentEnv

# Import the Hearthstone agents
from agents.RandomAgent import RandomAgent # RandomAgent implementation
from agents.HumanAgent import HumanAgent # HumanAgent implementation
from agents.DLAgents import DynamicLookaheadAgent # Dynamic Lookahead Agent implementation
from agents.GCAgents import GretiveCompAgent # Greedy Choice Agent implementation
from agents.SMAgents import NaiveScoreLookaheadAgent, WeightedScoreAgent # Sebastian Miller's agents
from agents.PPOAgent import PPOAgent # Proximal Policy Optimization (PPO) Agent implementation
from agents.GreedyAgent import GreedyAgent # GreedyAgent implementation
from agents.BaseDLAgent import BaseDynamicLookaheadAgent # BaseDynamicLookaheadAgent implementation

# Import the OSFP callback and average policy agent
from functions.OSFP import (
    OSFPCallback, AveragePolicyAgent, 
    AveragePolicyHolder, update_env_opponent
)

# Load configuration
import yaml
from yaml.loader import SafeLoader

with open('src/models/PPO/ppo_model_settings.yaml') as config_file:
    config = yaml.load(config_file, Loader=SafeLoader)
    model_config = config['new_model']

DEBUG = False

def mask_fn(env):
    """
    Function to generate action masks for the environment.
    
    :param env: The environment instance.
    :return: Action mask as a numpy array.
    """
    _, action_mask = env.hs_env.get_valid_actions()
    # action_mask is currently shape (102400,).

    # Double-check dtype
    action_mask = action_mask.astype(np.int64)

    return action_mask

def make_env(
    class1: int,
    class2: int,
    card_data: pd.DataFrame,
    opponent_agent: RandomAgent,
    final_reward_mode: int = None, 
    incremental_reward_mode: int = None, 
    embedded: bool = False,
    deck_include: bool = False,
    deck_include_v2: bool = False,
    deck1: list = None,
    deck2: list = None,
    class_options: dict = None,
    deck_metadata: pd.DataFrame = None,
    all_decks: pd.DataFrame = None,
    mirror_matches: bool = False,
    ) -> HearthVsAgentEnv:
    """
    Returns a new instance of HearthVsAgentEnv wrapped with ActionMasker.
    
    :param class1: Class of the first player.
    :param class2: Class of the second player.
    :param card_data: DataFrame containing card data.
    :param opponent_agent: The opponent agent to use.
    :param final_reward_mode: Mode for final reward calculation.
    :param incremental_reward_mode: Mode for incremental reward calculation.
    :param embedded: Whether to use embedded representation.
    :param deck_include: Whether to include deck information.
    :param deck_include_v2: Whether to include deck information in version 2.
    :param deck1: List of cards in the first player's deck.
    :param deck2: List of cards in the second player's deck.
    :param class_options: Dictionary mapping class indices to class names.
    :param deck_metadata: DataFrame containing deck metadata.
    :param all_decks: DataFrame containing all decks.
    :param mirror_matches: Whether to allow mirror matches.
    :return: A new instance of HearthVsAgentEnv wrapped with ActionMasker.
    """
    # 1) Create your base environment
    env = HearthVsAgentEnv(
        class1                  = class1, 
        class2                  = class2, 
        class_options           = class_options,
        card_data               = card_data, 
        opponent_agent          = opponent_agent,
        final_reward_mode       = final_reward_mode,
        incremental_reward_mode = incremental_reward_mode,
        embedded                = embedded,
        deck_include            = deck_include,
        deck_include_v2         = deck_include_v2,
        deck1                   = deck1,
        deck2                   = deck2,
        deck_metadata           = deck_metadata,
        all_decks               = all_decks,
        mirror_matches          = mirror_matches,
    )
    
    # 2) Wrap with ActionMasker if you need action masking
    env = ActionMasker(env, mask_fn)
    
    return env

def train_ppo_model():
    """
    Train a PPO model using the specified configuration.
    """
    
    # Load the card data from the csv file for the environment
    data_file = "src/data/card_data.csv"
    card_data = pd.read_csv(data_file)
    
    # Load the final decks for the environment
    decks = pd.read_csv('src/data/final_decks.csv')
    
    # Load available decks for the selected classes
    deck_metadata = pd.read_csv('src/data/decks_metadata.csv')
        
    print("Setting up environment...")
    # Create an instance of the opponent agent class and the classes for the player and opponent
    
    if model_config['opponent_agent'] == "RandomAgent":
        opponent_agent = RandomAgent()
    elif model_config['opponent_agent'] == "GreedyAgent":
        opponent_agent = GreedyAgent(score_method=model_config['opponent_method'])
    elif model_config['opponent_agent'] == "PPOAgent":
        opponent_agent = PPOAgent(model_path = model_config['opponent_method'], model_type = model_config['opponent_model_type'])
    else:
        opponent_agent = RandomAgent()
        
    class1 = model_config['player_class']
    deck1 = model_config['player_deck']
    
    class2 = model_config['opponent_class']
    deck2 = model_config['opponent_deck']
    
    final_reward_mode = model_config['final_reward_mode']
    incremental_reward_mode = model_config['incremental_reward_mode']
    
    embedded = model_config['embedded']
    deck_include = model_config['deck_include']
    deck_include_v2 = model_config['deck_include_v2']
        
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

    if class1 != "all":
        available_decks1 = deck_metadata[deck_metadata['hero_class'] == class_options[class1]]
        
        deck_id1 = available_decks1["deck_id"].values[deck1]
        
        deck1_list = decks[decks['deck_id'] == deck_id1]['card_id'].values
    else:
        deck1_list = "all"
    

    if class2 != "all":
        available_decks2 = deck_metadata[deck_metadata['hero_class'] == class_options[class2]]
        
        deck_id2 = available_decks2["deck_id"].values[deck2]
        
        deck2_list = decks[decks['deck_id'] == deck_id2]['card_id'].values
    else:
        deck2_list = "all"

    # Single process
    vec_env = DummyVecEnv([
        lambda: make_env(
            class1                  = class1, 
            class2                  = class2,
            card_data               = card_data,
            opponent_agent          = opponent_agent,
            final_reward_mode       = final_reward_mode, 
            incremental_reward_mode = incremental_reward_mode, 
            embedded                = embedded,
            deck_include            = deck_include,
            deck_include_v2         = deck_include_v2,
            deck1                   = deck1_list,
            deck2                   = deck2_list,
            class_options           = class_options,
            deck_metadata           = deck_metadata,
            all_decks               = decks,
            mirror_matches          = model_config['mirror_matches'],
            )
        ])
    
    # Monitor the environment
    vec_env = VecMonitor(vec_env, filename='logs/')

    # Define the policy network based on the model configuration
    policy_kwargs = dict(
        net_arch = dict(
            pi=model_config['policy_layers'],
            vf=model_config['value_layers']
        )
    )

    
    if model_config["device"] == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    if "previous_model" in model_config:
        previous_model = model_config['previous_model']
        
        print(f"Using previous model: {previous_model}")
        if model_config["model_type"] == "RNN":
            model = RecurrentPPO.load(previous_model, device=device)
        elif model_config["model_type"] == "MaskRNN":
            model = RecurrentMaskablePPO.load(previous_model, device=device)
        elif model_config["model_type"] == "Mask":
            model = MaskablePPO.load(previous_model, device=device)
        else:
            model = PPO.load(previous_model, device=device)
        
        model.set_env(vec_env)    
            
    else:
        # Create the model
        if model_config["model_type"] == "MaskRNN":
            model_type = RecurrentMaskablePPO
            model_policy = "MlpLstmPolicy"
        elif model_config["model_type"] == "RNN":
            model_type = RecurrentPPO
            model_policy = "MlpLstmPolicy"
        elif model_config["model_type"] == "Mask":
            model_type = MaskablePPO
            model_policy = "MlpPolicy"
        else:
            model_type = PPO
            model_policy = "MlpPolicy"
            
            
        if model_config['seed'] == "None":
            seed = None
        else:
            seed = model_config['seed']
            
        model = model_type(
            policy          = model_policy,
            policy_kwargs   = policy_kwargs,
            env             = vec_env, 
            device          = device,
            gamma           = model_config['gamma'],
            gae_lambda      = model_config['gae_lambda'],
            n_steps         = model_config['n_steps'],
            batch_size      = model_config['batch_size'],
            learning_rate   = model_config['learning_rate'],
            n_epochs        = model_config['n_epochs'],
            seed            = seed,
            clip_range      = model_config['clip_range'],
            ent_coef        = model_config['ent_coef'],
            verbose         = 1,
            tensorboard_log = model_config['tensorboard_log'],
        )
      
    # If total steps is larger than 100_000, set the save frequency to 50_000
    # If it is larger than 500_000, set the save frequency to 100_000
    if  model_config["total_steps"] > 1_000_000:
        save_freq = 250_000
    elif model_config["total_steps"] > 500_000:
        save_freq = 100_000
    elif model_config["total_steps"] > 100_000:
        save_freq = 50_000
    else:
        save_freq = 10_000
    
    # Create the callback: save every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,                 # how often (in timesteps) to save
        save_path="src/models/PPO/trained/checkpoints/",       # folder where the models will be saved
        name_prefix=f"{model_config["name"]}_checkpoint",   # prefix for saved model files
        
    )
        
    if model_config["opponent_agent"] == "OSFPAgent":
        # Create the AveragePolicyHolder
        average_holder = AveragePolicyHolder(
            current_model   = model,
            model_type      = model_type,
            model_policy    = model_policy,
            policy_kwargs   = policy_kwargs,
            env             = vec_env,
            device          = device,
            gamma           = model_config['gamma'],
            gae_lambda      = model_config['gae_lambda'],
            n_steps         = model_config['n_steps'],
            batch_size      = model_config['batch_size'],
            learning_rate   = model_config['learning_rate'],
            n_epochs        = model_config['n_epochs'],
            seed            = seed,
            clip_range      = model_config['clip_range'],
            ent_coef        = model_config['ent_coef']
        )
        
        # Create the OSFP callback using the holder
        osfp_callback = OSFPCallback(
            holder      = average_holder, 
            alpha       = model_config['osfp_alpha'], 
            update_freq = model_config['osfp_update_freq'], 
            verbose     = 1
            )
        
        # Create the opponent agent wrapper using the holder
        opponent_agent = AveragePolicyAgent(holder = average_holder)
        
        # Update the environment's opponent agent; assuming your vectorized env has this capability:
        update_env_opponent(vec_env, opponent_agent)
        
        # Ensure that your main model still uses the same environment
        model.set_env(vec_env)
        
        total_callbacks = [checkpoint_callback, osfp_callback]
    else:
        total_callbacks = [checkpoint_callback]
    
    print("Training model...")
    
    print("Model configuration:")
    # Print the configuration in a nice format
    print(json.dumps(model_config, indent=4))
    
    show_progress = True
    model.learn(
        model_config["total_steps"], 
        callback=total_callbacks, 
        progress_bar=show_progress
        )

    print("Done training")
    
    # Save the model
    model.save(f"src/models/PPO/trained/{model_config["name"]}.zip")
    print("Model saved successfully")
    
    # Run 
    print("Evaluating model against trained Opponent ...")
    eval_env = make_env(
            class1                  = class1, 
            class2                  = class2,
            card_data               = card_data,
            opponent_agent          = opponent_agent,
            final_reward_mode       = final_reward_mode, 
            incremental_reward_mode = incremental_reward_mode, 
            embedded                = embedded,
            deck_include            = deck_include,
            deck_include_v2         = deck_include_v2,
            deck1                   = deck1_list,
            deck2                   = deck2_list,
            class_options           = class_options,
            deck_metadata           = deck_metadata,
            all_decks               = decks,
            )
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=model_config["eval_episodes"], warn=False)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    print("Evaluating model against random agent ...")
    # Create a new random agent
    random_agent = RandomAgent()
    eval_env = make_env(
            class1                  = class1, 
            class2                  = class2,
            card_data               = card_data,
            opponent_agent          = random_agent,
            final_reward_mode       = final_reward_mode, 
            incremental_reward_mode = incremental_reward_mode, 
            embedded                = embedded,
            deck_include            = deck_include,
            deck_include_v2         = deck_include_v2,
            deck1                   = deck1_list,
            deck2                   = deck2_list,
            class_options           = class_options,
            deck_metadata           = deck_metadata,
            all_decks               = decks,
        )
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=model_config["eval_episodes"], warn=False)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
