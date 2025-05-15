import torch
import gymnasium
import numpy as np
import pandas as pd
import random 
import logging
import os
import json

# Import Fireplace modules
from fireplace.logging import log

from agents.PPOAgent import PPOAgent
from agents.GreedyAgent import GreedyAgent
from agents.RandomAgent import RandomAgent
from env.hearthstone.HearthVsAgentEnv import HearthVsAgentEnv

from models.EncodedPPO.MultiHeadAutoEncoder import MultiHeadAutoEncoder
from models.EncodedPPO.EncodedPPOEnv import make_env
from models.EncodedPPO.utils import (
    collect_training_data, train_autoencoder
)
from functions.data_loading import generate_feature_names

# Import the OSFP callback and average policy agent
from functions.OSFP import (
    OSFPCallback, AveragePolicyAgent, 
    AveragePolicyHolder, update_env_opponent
)

# Import the RecurrentMaskablePPO model
from models.PPO.recurrent_maskable.ppo_mask_recurrent import RecurrentMaskablePPO

# Stable Baselines
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

# Import evaluate_policy function
from stable_baselines3.common.evaluation import evaluate_policy

# Maskable PPO and other sb3_contrib imports
from sb3_contrib import MaskablePPO, RecurrentPPO

# Load configuration
import yaml
from yaml.loader import SafeLoader

with open('src/models/EncodedPPO/encoded_ppo_settings.yaml') as config_file:
    config = yaml.load(config_file, Loader=SafeLoader)
    model_config = config['new_model']

DEBUG = False
NAME = model_config['general']['name']
SAVE_FOLDER = os.path.join("src/models/EncodedPPO/save_folders/", NAME)
CONTROLLER_FILE = os.path.join(SAVE_FOLDER, model_config["controller"]['save_name'])
CONTROLLER_NAME = model_config["controller"]['save_name']

PRINT_WIDTH = 100

# Check if subdirectory exists, if not create it

if not os.path.exists(SAVE_FOLDER):
    raise FileNotFoundError(f"Model directory {SAVE_FOLDER} does not exist.")

def train_encoded_ppo():
    """
    Train the Encoded PPO for Hearthstone using a simulated environment.
    """
    
    # Set data collection flag
    # Set to True to collect training data from the real environment
    # Set to False to use pre-collected data
    COLLECT_DATA = model_config['general']['collect_data']
    TRAIN_AUTOENCODER = model_config['general']['train_encoder']
    TRAIN_CONTROLLER = model_config['general']['train_controller']
    
    ALL_STATES_PATH = os.path.join(SAVE_FOLDER, "all_states.npy")
    ENCODER_PATH = os.path.join(SAVE_FOLDER, "autoencoder.pth")
    
    print("=" * PRINT_WIDTH)
    print("Encoded PPO Configuration:")
    # Print the configuration in a nice format
    print(json.dumps(model_config, indent=4))
    print("=" * PRINT_WIDTH)
    
    log.setLevel(logging.ERROR)
    
    if model_config['general']['device'] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    ### SETUP ENVIRONMENT ###
    # Cache CSV data (read once)
    card_data = pd.read_csv("src/data/card_data.csv")
    decks = pd.read_csv('src/data/final_decks.csv')
    deck_metadata = pd.read_csv('src/data/decks_metadata.csv')
    
    # Precompute available decks by hero class
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
    card_classes = [None, None]
    
    # Get available decks for each class
    available_decks = {}
    for cls in class_options.values():
        available_decks[cls] = deck_metadata[deck_metadata['hero_class'] == cls]
    
    # Select decks and classes
    deck1 = model_config['data_collection']['deck1']
    deck2 = model_config['data_collection']['deck2']
    class1 = model_config['data_collection']['class1']
    class2 = model_config['data_collection']['class2']

    # Dynamic class selection if "all" is chosen
    if class1 == "all":
        selected_class1 = random.choice(list(class_options.keys()))
        card_classes[0] = class_options[selected_class1]
    if class2 == "all":
        selected_class2 = random.choice(list(class_options.keys()))
        card_classes[1] = class_options[selected_class2]
        
        # Get deck lists based on precomputed available_decks
        if deck1 == "all":
            available = available_decks[card_classes[0]]
            selected_deck1 = random.choice(range(len(available)))
        else:
            selected_deck1 = deck1
        deck_id1 = available_decks[card_classes[0]]["deck_id"].values[selected_deck1]
        deck1_list = decks[decks['deck_id'] == deck_id1]['card_id'].values
    
        if deck2 == "all":
            available = available_decks[card_classes[1]]
            selected_deck2 = random.choice(range(len(available)))
        else:
            selected_deck2 = deck2
        deck_id2 = available_decks[card_classes[1]]["deck_id"].values[selected_deck2]
        deck2_list = decks[decks['deck_id'] == deck_id2]['card_id'].values
    
    EMBEDDED = model_config['general']['embedded']
    DECK_INCLUDE = model_config['general']['deck_include']
    DECK_INCLUDE_V2 = model_config['general']['deck_include_v2']
    
    SAMPLING_AGENT = model_config['data_collection']['sampling_agent']
    SCORE_METHOD = model_config['data_collection']['score_method']
    
    # Get the feature names and indices for continuous and discrete features
    feature_names, cont_indices, disc_indices = generate_feature_names(
        embedded        = EMBEDDED, 
        deck_include    = DECK_INCLUDE,
        deck_include_v2 = DECK_INCLUDE_V2,
        embedding_size  = 368 # size of the embedding (minilml6 used, so 368)
    )
    
    # Initialize environment once per game (passes pre-read card_data)
    env = gymnasium.make(
        id                      = "hearthstone_env/HearthGym-v0",
        class1                  = selected_class1,
        class2                  = selected_class2,
        clone                   = None,
        card_data               = card_data,
        final_reward_mode       = model_config['data_collection']['final_reward_mode'],
        incremental_reward_mode = model_config['data_collection']['incremental_reward_mode'],
        embedded                = EMBEDDED,
        deck_include            = DECK_INCLUDE,
        deck_include_v2         = DECK_INCLUDE_V2,
        deck1                   = deck1_list,
        deck2                   = deck2_list,
        class_options           = class_options,
    ).unwrapped
    
    if COLLECT_DATA:
        DATA_EPISODES = model_config['data_collection']['num_episodes']
        
        # 1. Collect training data.
        print("Collecting training data from real env...")
        data = collect_training_data(
            env=env, 
            num_episodes=DATA_EPISODES, 
            sampling_agent=SAMPLING_AGENT,
            score_method=SCORE_METHOD
            )    
        
        print("Flattening states...")
        # Flatten states from all episodes. 
        all_states = np.array([
            obs for episode in data for (obs, _, _, _, _) in episode
            ])
        
        print("Saving results...")
        np.save(ALL_STATES_PATH, all_states)
        
        print("All collected data saved.")
        
    print("-" * PRINT_WIDTH)
    print("Loading training data...")
    
    # Load data (pkl) and all_states (npy) for later use (only if used)
    if TRAIN_AUTOENCODER:
        all_states = np.load(ALL_STATES_PATH)
    
    cont_input_dim = len(cont_indices)  # number of continuous features
    disc_input_dim = len(disc_indices)   # number of discrete features
    latent_dim = model_config["encoder"]["latent_dim"]  # chosen latent size
    cont_hidden_dim = model_config["encoder"]["cont_hidden_dim"]
    disc_hidden_dim = model_config["encoder"]["disc_hidden_dim"]
    
    if TRAIN_AUTOENCODER:
        autoencoder = MultiHeadAutoEncoder(
            cont_input_dim=cont_input_dim, 
            disc_input_dim=disc_input_dim, 
            latent_dim=latent_dim,
            cont_hidden_dim=cont_hidden_dim,
            disc_hidden_dim=disc_hidden_dim
            ).to(device)
        
        ENCODER_EPOCHS = model_config['encoder']['epochs']
        BATCH_SIZE = model_config['encoder']['batch_size']
        LEARNING_RATE = model_config['encoder']['learning_rate']
        
        # 2. Train the autoencoder.
        train_autoencoder(
            autoencoder=autoencoder, 
            training_states=all_states, 
            num_epochs=ENCODER_EPOCHS, 
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            device=device,
            disc_indices=disc_indices,
            cont_indices=cont_indices,
            )

        # Save autoencoder
        torch.save({
            'cont_input_dim': cont_input_dim,
            'disc_input_dim': disc_input_dim,
            'latent_dim':     latent_dim,
            'cont_hidden_dim': cont_hidden_dim,
            'disc_hidden_dim': disc_hidden_dim,
            'autoencoder_state_dict': autoencoder.state_dict()
        }, ENCODER_PATH)
        print("Autoencoder trained and saved.")
        
    # Load autoencoder
    ckpt = torch.load(ENCODER_PATH, map_location=device, weights_only=True)
    autoencoder = MultiHeadAutoEncoder(
        cont_input_dim = ckpt['cont_input_dim'],
        disc_input_dim = ckpt['disc_input_dim'],
        latent_dim     = ckpt['latent_dim'],
        cont_hidden_dim=ckpt['cont_hidden_dim'],
        disc_hidden_dim=ckpt['disc_hidden_dim']
    ).to(device)
    
    autoencoder.load_state_dict(ckpt['autoencoder_state_dict'])
    autoencoder.eval()
    
    print("Autoencoder loaded.")
    print("-" * PRINT_WIDTH)
    
    opponent_agent = RandomAgent()
    if model_config["controller"]['opponent_agent'] == "RandomAgent":
        opponent_agent = RandomAgent()
    elif model_config["controller"]['opponent_agent'] == "GreedyAgent":
        opponent_agent = GreedyAgent(score_method=model_config["controller"]['opponent_method'])
    elif model_config["controller"]['opponent_agent'] == "PPOAgent":
        opponent_agent = PPOAgent(model_path = model_config["controller"]['opponent_method'], model_type = model_config["controller"]['opponent_model_type'])
    else:
        opponent_agent = RandomAgent()
    
    
    # Select decks and classes
    class1 = model_config['controller']['player_class']
    deck1 = model_config['controller']['player_deck']
    class2 = model_config['controller']['opponent_class']
    deck2 = model_config['controller']['opponent_deck']

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
    
    real_env = HearthVsAgentEnv(
        class1                  = class1, 
        class2                  = class2,
        class_options           = class_options,
        card_data               = card_data, 
        opponent_agent          = opponent_agent,
        final_reward_mode       = model_config['controller']['final_reward_mode'],
        incremental_reward_mode = model_config['controller']['incremental_reward_mode'],
        embedded                = EMBEDDED,
        deck_include            = DECK_INCLUDE,
        deck_include_v2         = DECK_INCLUDE_V2,
        deck1                   = deck1_list,
        deck2                   = deck2_list,
        deck_metadata           = deck_metadata,
        all_decks               = decks,
        mirror_matches          = model_config["controller"]['mirror_matches'],
    )
    
    # Observation shape is output shape of the autoencoder
    observation_shape = (autoencoder.latent_dim,)
    
    sim_env = DummyVecEnv([
        lambda: make_env(
            autoencoder         = autoencoder,
            real_env            = real_env,
            device              = device,
            cont_indices        = cont_indices,
            disc_indices        = disc_indices,
            observation_shape   = observation_shape,
            )
        ])
    
    # Monitor the environment
    vec_env = VecMonitor(sim_env, filename=os.path.join(SAVE_FOLDER, CONTROLLER_NAME))

    if TRAIN_CONTROLLER:
        # 5. Train a controller (via PPO) in the simulated env.
        print("Training controller on simulated env...")
        
        # Define the policy network based on the model configuration
        policy_kwargs = dict(
            net_arch = dict(
                pi=model_config["controller"]['policy_layers'],
                vf=model_config["controller"]['value_layers']
            )
        )
        
        if "previous_model" in model_config["controller"]:
            previous_model = model_config["controller"]['previous_model']
            
            print(f"Using previous model: {previous_model}")
            if model_config["controller"]["model_type"] == "RNN":
                controller_model = RecurrentPPO.load(previous_model, device=device)
            elif model_config["controller"]["model_type"] == "MaskRNN":
                controller_model = RecurrentMaskablePPO.load(previous_model, device=device)
            elif model_config["controller"]["model_type"] == "Mask":
                controller_model = MaskablePPO.load(previous_model, device=device)
            else:
                controller_model = PPO.load(previous_model, device=device)
            
            controller_model.set_env(vec_env)    
                
        else:
            # Create the model
            if model_config["controller"]["model_type"] == "MaskRNN":
                model_type = RecurrentMaskablePPO
                model_policy = "MlpLstmPolicy"
            elif model_config["controller"]["model_type"] == "RNN":
                model_type = RecurrentPPO
                model_policy = "MlpLstmPolicy"
            elif model_config["controller"]["model_type"] == "Mask":
                model_type = MaskablePPO
                model_policy = "MlpPolicy"
            else:
                model_type = PPO
                model_policy = "MlpPolicy"
                
                
            if model_config["controller"]['seed'] == "None":
                seed = None
            else:
                seed = model_config["controller"]['seed']
                
            controller_model = model_type(
                policy          = model_policy,
                policy_kwargs   = policy_kwargs,
                env             = vec_env, 
                device          = device,
                gamma           = model_config["controller"]['gamma'],
                gae_lambda      = model_config["controller"]['gae_lambda'],
                n_steps         = model_config["controller"]['n_steps'],
                batch_size      = model_config["controller"]['batch_size'],
                learning_rate   = model_config["controller"]['learning_rate'],
                n_epochs        = model_config["controller"]['n_epochs'],
                seed            = seed,
                clip_range      = model_config["controller"]['clip_range'],
                ent_coef        = model_config["controller"]['ent_coef'],
                verbose         = 1,
                tensorboard_log = model_config["controller"]['tensorboard_log'],
            )
        
        # If total steps is larger than 100_000, set the save frequency to 50_000
        # If it is larger than 500_000, set the save frequency to 100_000
        if  model_config["controller"]["total_steps"] > 1_000_000:
            save_freq = 200_000
        elif model_config["controller"]["total_steps"] > 500_000:
            save_freq = 100_000
        elif model_config["controller"]["total_steps"] > 100_000:
            save_freq = 50_000
        else:
            save_freq = 10_000
        
        # Create the callback: save every 'save_freq' steps
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,                 # how often (in timesteps) to save
            save_path=os.path.join(SAVE_FOLDER, "controller_checkpoints"),  # directory to save the model
            name_prefix=f"{CONTROLLER_NAME}_checkpoint",   # prefix for saved model files
        )
            
        if model_config["controller"]["opponent_agent"] == "OSFPAgent":
            # Create the AveragePolicyHolder
            average_holder = AveragePolicyHolder(
                current_model   = controller_model,
                model_type      = model_type,
                model_policy    = model_policy,
                policy_kwargs   = policy_kwargs,
                env             = vec_env,
                device          = device,
                gamma           = model_config["controller"]['gamma'],
                gae_lambda      = model_config["controller"]['gae_lambda'],
                n_steps         = model_config["controller"]['n_steps'],
                batch_size      = model_config["controller"]['batch_size'],
                learning_rate   = model_config["controller"]['learning_rate'],
                n_epochs        = model_config["controller"]['n_epochs'],
                seed            = seed,
                clip_range      = model_config["controller"]['clip_range'],
                ent_coef        = model_config["controller"]['ent_coef']
            )
            
            # Create the OSFP callback using the holder
            osfp_callback = OSFPCallback(
                holder      = average_holder, 
                alpha       = model_config["controller"]["osfp_alpha"], 
                update_freq = model_config["controller"]['osfp_update_freq'], 
                verbose     = 1
                )
            
            # Create the opponent agent wrapper using the holder
            opponent_agent = AveragePolicyAgent(average_holder)
            
            # Update the environment's opponent agent; assuming your vectorized env has this capability:
            update_env_opponent(vec_env, opponent_agent)
            
            # Ensure that your main model still uses the same environment
            controller_model.set_env(vec_env)
            
            total_callbacks = [checkpoint_callback, osfp_callback]
        else:
            total_callbacks = [checkpoint_callback]
        
        print("Training Controller Model...")        
        show_progress = True
        controller_model.learn(
            model_config["controller"]["total_steps"], 
            callback=total_callbacks, 
            progress_bar=show_progress
            )

        print("Done training")

        controller_model.save(os.path.join(SAVE_FOLDER, f"{CONTROLLER_NAME}.zip"))
        print("Model saved successfully")

    eval_env = vec_env
    eval_env.reset()
    
     # Load the trained model    
    if model_config["controller"]["model_type"] == "RNN":
        controller_model = RecurrentPPO.load(os.path.join(SAVE_FOLDER, f"{CONTROLLER_NAME}.zip"), device=device)
    elif model_config["controller"]["model_type"] == "MaskRNN":
        controller_model = RecurrentMaskablePPO.load(os.path.join(SAVE_FOLDER, f"{CONTROLLER_NAME}.zip"), device=device)
    elif model_config["controller"]["model_type"] == "Mask":
        controller_model = MaskablePPO.load(os.path.join(SAVE_FOLDER, f"{CONTROLLER_NAME}.zip"), device=device)
    else:
        controller_model = PPO.load(os.path.join(SAVE_FOLDER, f"{CONTROLLER_NAME}.zip"), device=device)
        
    controller_model.set_env(eval_env)
    
    print("Evaluating model...")
    # Evaluate the controller on the real environment
    episode_rewards, episode_lengths = evaluate_policy(
        controller_model, 
        eval_env, 
        n_eval_episodes         = model_config["controller"]['eval_episodes'], 
        warn                    = False,
        deterministic           = False,
        return_episode_rewards  = True,
        )
    
    # Get mean and std of rewards
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    # Print all rewards
    print("All rewards:")
    print(episode_rewards)
