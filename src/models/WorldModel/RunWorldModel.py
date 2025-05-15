import torch
import gymnasium
import numpy as np
import pandas as pd
import random 
import logging
import pickle
import os
import json
from torch.utils.data import DataLoader

# Import Fireplace modules
from fireplace.logging import log

from env.hearthstone.HearthVsAgentEnv import HearthVsAgentEnv

from models.WorldModel.RSSM import RSSM
from models.WorldModel.MultiHeadAutoEncoder import MultiHeadAutoEncoder
from models.WorldModel.LegalityNet import LegalityNet
from models.WorldModel.DreamHearthEnv import make_env
from models.WorldModel.utils import (
    collect_training_data, train_autoencoder, train_rssm, train_legality
)
from functions.data_loading import generate_feature_names
from models.WorldModel.DreamerCallback import DreamerV3MaskableCallback
from models.WorldModel.TwoHotMaskable import TwoHotMaskablePPO
from models.WorldModel.RSSMSequenceDataset import RSSMSequenceDataset
from models.WorldModel.LegalityDataset import LegalitySequenceDataset

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

with open('src/models/WorldModel/world_model_settings.yaml') as config_file:
    config = yaml.load(config_file, Loader=SafeLoader)
    model_config = config['new_model']

DEBUG = False
NAME = model_config['general']['name']
SAVE_FOLDER = os.path.join("src/models/WorldModel/save_folders/", NAME)

PRINT_WIDTH = 100

# Check if subdirectory exists, if not create it

if not os.path.exists(SAVE_FOLDER):
    raise FileNotFoundError(f"Model directory {SAVE_FOLDER} does not exist.")

def train_world_model():
    """
    Train the world model for Hearthstone using a simulated environment.
    """
    
    # Set data collection flag
    # Set to True to collect training data from the real environment
    # Set to False to use pre-collected data
    COLLECT_DATA = model_config['general']['collect_data']
    COLLECT_MASKS = model_config['general']['collect_masks']
    TRAIN_AUTOENCODER = model_config['general']['train_encoder']
    TRAIN_RSSM = model_config['general']['train_rssm']
    TRAIN_LEGALITY = model_config['general']['train_legality']
    TRAIN_CONTROLLER = model_config['general']['train_controller']
    
    TRAINING_DATA_PATH = os.path.join(SAVE_FOLDER, "training_data.pkl")
    MASK_PATH = os.path.join(SAVE_FOLDER, "mask_data.npz")
    ALL_STATES_PATH = os.path.join(SAVE_FOLDER, "all_states.npy")
    ENCODER_PATH = os.path.join(SAVE_FOLDER, "autoencoder.pth")
    RSSM_PATH = os.path.join(SAVE_FOLDER, "rssm.pth")
    LEGALITY_PATH = os.path.join(SAVE_FOLDER, "legality_net.pth")
    
    print("=" * PRINT_WIDTH)
    print("World Model Configuration:")
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
        card_data               =card_data,
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
        # data, mask_data = collect_training_data(env, num_episodes=DATA_EPISODES, collect_masks=COLLECT_MASKS)
        data, mask_data = collect_training_data(
            env=env, 
            num_episodes=DATA_EPISODES, 
            collect_masks=COLLECT_MASKS, 
            sampling_agent=SAMPLING_AGENT,
            score_method=SCORE_METHOD
            )
        
        print("Saving training data...")
        # Save data (pkl) and all_states (npy) for later use
        with open(TRAINING_DATA_PATH, 'wb') as f:
            pickle.dump(data, f)
            
        if COLLECT_MASKS:
            print("Saving Mask data...")
            np.savez_compressed(MASK_PATH, mask_data=mask_data)
        
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
        with open(TRAINING_DATA_PATH, 'rb') as f:
            data = pickle.load(f)
    else:
        if TRAIN_RSSM or TRAIN_LEGALITY:
            with open(TRAINING_DATA_PATH, 'rb') as f:
                data = pickle.load(f)
        
    if TRAIN_LEGALITY:
        loaded = np.load(MASK_PATH, allow_pickle=True)
        mask_data = loaded["mask_data"].tolist()
    
    cont_input_dim = len(cont_indices)  # number of continuous features
    disc_input_dim = len(disc_indices)   # number of discrete features
    
    num_cats   = model_config['world_model']['num_cats']          # e.g. 32
    cat_dim    = model_config['world_model']['cat_dim']           # e.g. 32
    latent_dim = num_cats * cat_dim
    
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

    # Create world model object.   
    if TRAIN_RSSM:        
        rssm = RSSM(
            cat_dim=cat_dim,
            num_cats=num_cats,
            hidden_dim  = model_config['rssm']['hidden_dim'],
            action_dim  = env.action_space.n
        ).to(device)

        world_model = type("WorldModel", (), {})()
        world_model.autoencoder = autoencoder
        world_model.rssm     = rssm

        print("-" * PRINT_WIDTH)
        print("Preparing training data for RSSM...")
        dataset = RSSMSequenceDataset(
            episodes        = data,
            world_model     = world_model,
            sequence_length = model_config['rssm']['sequence_length'],
            cont_indices    = cont_indices,
            disc_indices    = disc_indices,
            num_actions     = env.action_space.n,
            device          = device
        )
        loader = DataLoader(
            dataset,
            batch_size = model_config['rssm']['batch_size'],
            shuffle    = False,           # no need to shuffle an IterableDataset
            num_workers=0,                # avoid extra worker memory :contentReference[oaicite:1]{index=1}
            pin_memory = False            # pinning here only affects CPU→GPU copy, but dataset already on GPU
        )

        print("Training RSSM...")
        train_rssm(
            model         = rssm,
            dataloader    = loader,
            epochs        = model_config['rssm']['epochs'],
            lr            = model_config['rssm']['learning_rate'],
            device        = device,
            print_every   = model_config['rssm']['print_every']
        )

        # Save RSSM
        torch.save({
            'cat_dim':         cat_dim,
            'num_cats':        num_cats,
            'action_dim':      env.action_space.n,
            'hidden_dim':      model_config['rssm']['hidden_dim'],
            'state_dict':      rssm.state_dict()
        }, RSSM_PATH)
        print("RSSM trained and saved.")
    
    # Load RSSM
    ckpt = torch.load(RSSM_PATH, map_location=device, weights_only=False)
    rssm = RSSM(
            cat_dim     = ckpt['cat_dim'],
            num_cats    = ckpt['num_cats'],
            hidden_dim  = ckpt['hidden_dim'],
            action_dim  = ckpt['action_dim'],
        ).to(device)
    rssm.load_state_dict(ckpt['state_dict'])
    rssm.eval()
    
    print("RSSM loaded.")
    print("-" * PRINT_WIDTH)
    
    world_model = type('WorldModel', (), {})()
    world_model.autoencoder = autoencoder
    world_model.rssm = rssm

    if TRAIN_LEGALITY:
        print("Training legality network...")
        action_dim = env.action_space.n

        legality_net = LegalityNet(
            in_dim     = latent_dim + world_model.rssm.hidden_dim,
            n_actions  = action_dim,
            widths     = model_config['legality_net']['widths'],
        ).to(device)

        leg_ds = LegalitySequenceDataset(
            mask_data      = mask_data,          # your list of episodes of (obs,act,mask)
            world_model    = world_model,        # with .autoencoder & .rssm loaded
            num_actions    = action_dim,
            cont_indices   = cont_indices,
            disc_indices   = disc_indices,
            device         = device,
        )
        
        loader = DataLoader(
            leg_ds,
            batch_size = model_config['legality_net']['batch_size'],
            shuffle    = False,           # no need to shuffle an IterableDataset
            num_workers=0,                # avoid extra worker memory :contentReference[oaicite:1]{index=1}
            pin_memory = False            # pinning here only affects CPU→GPU copy, but dataset already on GPU
        )
        
        leg_net = train_legality(
            dataloader=loader,
            legality_net = legality_net,
            epochs  = model_config['legality_net']['epochs'],
            lr      = model_config['legality_net']['learning_rate'],
            device  = device,
        )

        # Save
        torch.save({
            'state_dict': leg_net.state_dict(),
            'in_dim'     : latent_dim + world_model.rssm.hidden_dim,
            'n_actions'  : action_dim,
            'widths'     : model_config['legality_net']['widths'],
        }, LEGALITY_PATH)

        print("Legality net trained and saved.")
        
    # Load legality net
    ckpt = torch.load(LEGALITY_PATH, map_location=device, weights_only=False)
    legality_net = LegalityNet(
        in_dim = ckpt['in_dim'],
        n_actions = ckpt['n_actions'],
        widths = ckpt['widths'],
    ).to(device)
    legality_net.load_state_dict(ckpt['state_dict'])
    legality_net.eval()  
    
    world_model.legality_net = legality_net      

    from agents.RandomAgent import RandomAgent        
    
    opponent_agent = RandomAgent()
    
    real_env = HearthVsAgentEnv(
        class1                  = class1, 
        class2                  = class2, 
        class_options           = class_options,
        card_data               = card_data, 
        opponent_agent          = opponent_agent,
        final_reward_mode       = 2,
        incremental_reward_mode = 0,
        embedded                = EMBEDDED,
        deck_include            = DECK_INCLUDE,
        deck_include_v2         = DECK_INCLUDE_V2,
        deck1                   = deck1,
        deck2                   = deck2,
        deck_metadata           = deck_metadata,
        all_decks               = decks,
        mirror_matches          = model_config["controller"]['mirror_matches'],
    )
    
    # The observation shape is the hidden state of the autoencoder + the hidden state of the RSSM
    observation_shape = (latent_dim + world_model.rssm.hidden_dim,)
    sim_env = DummyVecEnv([
        lambda: make_env(
            world_model         = world_model, 
            real_env            = real_env,
            device              = device,
            cont_indices        = cont_indices,
            disc_indices        = disc_indices,
            observation_shape   = observation_shape,
            max_steps           = model_config["controller"]['max_steps'],
            )
        ])
    
    # Monitor the environment
    vec_env = VecMonitor(sim_env, filename=SAVE_FOLDER)

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
                
        if model_config["controller"]["model_type"] == "TwoHotMask":
            # Use our TwoHotMaskablePolicy
            controller_model = TwoHotMaskablePPO(
                policy  = "MlpPolicy",
                env=vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device,
                gamma=model_config["controller"]['gamma'],
                gae_lambda=model_config["controller"]['gae_lambda'],
                n_steps=model_config["controller"]['n_steps'],
                batch_size=model_config["controller"]['batch_size'],
                learning_rate=model_config["controller"]['learning_rate'],
                n_epochs=model_config["controller"]['n_epochs'],
                clip_range=model_config["controller"]['clip_range'],
                ent_coef=model_config["controller"]['ent_coef'],
                seed=seed,
                return_bins=world_model.rssm.return_bins,
            )     
        else: 
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
        
        print("Controller model type: ", model_type)
        print("Controller model policy: ", model_policy)
        print("Controller model kwargs: ", policy_kwargs)
        print("Return bins: ", world_model.rssm.return_bins)
        
        
        
        # If total steps is larger than 100_000, set the save frequency to 50_000
        # If it is larger than 500_000, set the save frequency to 100_000
        if model_config["controller"]["total_steps"] > 10_000_000:
            save_freq = 2_000_000
        elif  model_config["controller"]["total_steps"] > 1_000_000:
            save_freq = 250_000
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
            name_prefix=f"{NAME}_checkpoint",   # prefix for saved model files
            
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
        else:
            osfp_callback = None
            
        if model_config["controller"]["use_dreamer_callback"]:
            dream_callback = DreamerV3MaskableCallback(
                checkpoint_callback=checkpoint_callback,
                osfp_callback=osfp_callback,
                sigma=model_config["controller"]["ema_sigma"],
                return_percentile=model_config["controller"]["return_percentile"],
            )
            final_callback = dream_callback
        else:
            final_callback = [checkpoint_callback]
            if osfp_callback is not None:
                final_callback.append(osfp_callback)
        
        print("Training Controller Model...")        
        show_progress = True
        controller_model.learn(
            model_config["controller"]["total_steps"], 
            callback=final_callback, 
            progress_bar=show_progress
            )

        print("Done training")

        controller_model.save(os.path.join(SAVE_FOLDER, f"{NAME}.zip"))
        print("Model saved successfully")

    eval_env = vec_env
    eval_env.reset()
    
     # Load the trained model    
    if model_config["controller"]["model_type"] == "RNN":
        controller_model = RecurrentPPO.load(os.path.join(SAVE_FOLDER, f"{NAME}.zip"), device=device)
    elif model_config["controller"]["model_type"] == "MaskRNN":
        controller_model = RecurrentMaskablePPO.load(os.path.join(SAVE_FOLDER, f"{NAME}.zip"), device=device)
    elif model_config["controller"]["model_type"] == "Mask":
        controller_model = MaskablePPO.load(os.path.join(SAVE_FOLDER, f"{NAME}.zip"), device=device)
    else:
        controller_model = PPO.load(os.path.join(SAVE_FOLDER, f"{NAME}.zip"), device=device)
        
    controller_model.set_env(eval_env)
    
    # Run 
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
