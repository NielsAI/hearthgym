# README for creating Encoded PPO agents

This repository contains a framework for creating EncodedPPO agents for the Hearthstone game. The framework is based on the `fireplace` package, which is a Python implementation of the Hearthstone game. The framework is designed to be used with reinforcement learning algorithms to train agents to play Hearthstone.

## Setup
The main component to configure when creating a new agent is the `src/models/EncodedPPO/encoded_ppo_settings.yaml` file. This file has multiple key entries, the first being `template`, showing all required parameters for creating a new agent. The new agent configuration should be stated with the key `new_model`, this is then used by the script. Other entries can be stored in this file to keep track of previously created models. Below are the required parameters for creating a new agent and their explanations.


<details>
<summary>Click to expand the YAML code</summary>

```yaml
new_model: # Template for modelling
  general: # General settings
    name: SAVE_FOLDER_NAME
    device: cuda
    embedded: False
    deck_include: False
    deck_include_v2: False
    collect_data: True
    train_encoder: True
    train_controller: True
  data_collection: # Data collection settings
    num_episodes: 1000
    final_reward_mode: 2 # 0: 10 for win and -10 for loss, 1: 100 for win and -100 for loss, else: 1 for win and -1 for loss
    incremental_reward_mode: 0
    class1: all
    class2: all
    deck1: all
    deck2: all
    sampling_agent: RandomAgent
    score_method: None
  encoder: # MultiHeadAutoEncoder settings
    epochs: 50
    batch_size: 128
    learning_rate: 0.001
    latent_dim: 50
    cont_hidden_dim: 128
    disc_hidden_dim: 32
  controller: # Controller settings (PPO-based)
    save_name: PPO_MODEL_NAME
    model_type: Mask
    policy_layers: [512, 512]
    value_layers: [512, 512]
    learning_rate: 0.00001
    gamma: 0.99
    gae_lambda: 0.95
    n_steps: 512
    batch_size: 256
    clip_range: 0.2
    ent_coef: 0.0
    n_epochs: 5
    seed: 42
    total_steps: 1_000_000
    eval_episodes: 5
    final_reward_mode: 2 # 0: 10 for win and -10 for loss, 1: 100 for win and -100 for loss, else: 1 for win and -1 for loss
    incremental_reward_mode: 0
    player_class: all
    player_deck: all
    opponent_class: all
    opponent_deck: all
    mirror_matches: False
    opponent_agent: RandomAgent
    opponent_method: None
    osfp_alpha: None
    osfp_update_freq: None
    tensorboard_log: TENSORBOARD_FOLDER
```

</details>


<details>
<summary>Click to expand the parameter explanation</summary>

The configuration file is divided into several sections, each containing different parameters: `general`, `data_collection`, `encoder`, and `controller`. Each section contains parameters that control the behavior of the agent. The most important parameters are described below:
- `general`: General settings for the agent. 
    - `name`: The `name` parameter specifies the name of the agent. 
    - `device`: The `device` parameter specifies the device to use for training (e.g., `cuda` or `cpu`). 
    - `embedded`: The `embedded` parameter specifies whether to use an embedded model or not. 
    - `deck_include`/`deck_include_v2`: The `deck_include` and `deck_include_v2` parameters specify whether to include the deck in the state representation. 
    - `collect_data`: The `collect_data` parameter specifies whether to collect data during training. 
    - `collect_masks`: The `collect_masks` parameter specifies whether to collect masks during training. 
    - `train_*`: The `train_encoder` and `train_controller` parameters specify whether to train the encoder and controller respectively.
- `data_collection`:
    - `num_episodes`: The `num_episodes` parameter specifies the number of episodes to collect data for. 
    - `final_reward_mode`: The `final_reward_mode` parameter specifies the reward mode for the final reward. 
    - `incremental_reward_mode`: The `incremental_reward_mode` parameter specifies the reward mode for the incremental reward. 
    - `class1`/`class2`: The `class1` and `class2` parameters specify the classes to use for data collection. 
    - `deck1`/`deck2`: The `deck1` and `deck2` parameters specify the decks to use for data collection. 
    - `sampling_agent`: The `sampling_agent` parameter specifies the agent to use for sampling. 
    - `score_method`: The `score_method` parameter specifies the method to use for scoring.
- `encoder`: The `encoder` section contains parameters for the encoder.
    - `epochs`: The `epochs` parameter specifies the number of epochs to train the encoder. 
    - `batch_size`: The `batch_size` parameter specifies the batch size to use for training. 
    - `learning_rate`: The `learning_rate` parameter specifies the learning rate to use for training. 
    - `latent_dim`: The `latent_dim` parameter specifies the latent dimension of the encoder. 
    - `cont_hidden_dim`: The `cont_hidden_dim` parameter specifies the hidden dimension of the controller. 
    - `disc_hidden_dim`: The `disc_hidden_dim` parameter specifies the hidden dimension of the discriminator.
- `controller`: The `controller` section contains parameters for the controller.
    - `save_name`: The `save_name` parameter specifies the name of the controller. 
    - `model_type`: The `model_type` parameter specifies the type of model to use (e.g., `Mask`). 
    - `policy_layers`: The `policy_layers` parameter specifies the layers for the policy network. 
    - `value_layers`: The `value_layers` parameter specifies the layers for the value network. 
    - `learning_rate`: The `learning_rate` parameter specifies the learning rate to use for training. 
    - `gamma`: The `gamma` parameter specifies the discount factor to use for training. 
    - `gae_lambda`: The `gae_lambda` parameter specifies the GAE lambda to use for training. 
    - `n_steps`: The `n_steps` parameter specifies the number of steps to use for training. 
    - `batch_size`: The `batch_size` parameter specifies the batch size to use for training. 
    - `clip_range`: The `clip_range` parameter specifies the clip range to use for training. 
    - `ent_coef`: The `ent_coef` parameter specifies the entropy coefficient to use for training. 
    - `n_epochs`: The `n_epochs` parameter specifies the number of epochs to use for training. 
    - `seed`: The seed used for random number generation.
    - `total_steps`: The total number of steps to train the agent.
    - `eval_episodes`: The number of episodes to evaluate the agent.
    - Other parameters are similar to those in the encoder section.
    - `final_reward_mode`: The `final_reward_mode` parameter specifies the reward mode for the final reward.
    - `incremental_reward_mode`: The `incremental_reward_mode` parameter specifies the reward mode for the incremental reward.
    - `player_class`/`player_deck`: The `player_class` and `player_deck` parameters specify the class and deck to use for the player. (See the list of classes and decks in the `src/run_game.py` file or general README.md)
    - `opponent_class`/`opponent_deck`: The `opponent_class` and `opponent_deck` parameters specify the class and deck to use for the opponent. (See the list of classes and decks in the `src/run_game.py` file or general README.md)
    - `mirror_matches`: The `mirror_matches` parameter specifies whether to use mirror matches or not.
    - `opponent_agent`: The `opponent_agent` parameter specifies the agent to use for the opponent.
    - `opponent_method`: The `opponent_method` parameter specifies the method to use for the opponent.
    - `osfp_alpha`: The `osfp_alpha` parameter specifies the alpha value to use for the OSFP algorithm.
    - `osfp_update_freq`: The `osfp_update_freq` parameter specifies the update frequency for the OSFP algorithm.
    - `tensorboard_log`: The `tensorboard_log` parameter specifies the folder to use for TensorBoard logs.
    

</details>

## Training
To train the agent, run the following command:
```bash
python src/train_encoded_ppo.py
```

It automatically detects the new model in the `encoded_ppo_settings.yaml` file and starts training it. The training process will save the model checkpoints and logs in the specified folders.