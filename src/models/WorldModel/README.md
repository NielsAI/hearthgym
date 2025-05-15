# README for creating World Model agents

This repository contains a framework for creating World Model agents for the Hearthstone game. The framework is based on the `fireplace` package, which is a Python implementation of the Hearthstone game. The framework is designed to be used with reinforcement learning algorithms to train agents to play Hearthstone.

## Setup
The main component to configure when creating a new agent is the `src/models/WorldModel/world_model_settings.yaml` file. This file has multiple key entries, the first being `template`, showing all required parameters for creating a new agent. The new agent configuration should be stated with the key `new_model`, this is then used by the script. Other entries can be stored in this file to keep track of previously created models. Below are the required parameters for creating a new agent and their explanations.


<details>
<summary>Click to expand the YAML code</summary>

```yaml
new_model: 
  general: # General settings
    name: SAVE_FOLDER_NAME
    device: cuda
    embedded: False
    deck_include: False
    deck_include_v2: False
    collect_data: True
    collect_masks: True
    train_encoder: True
    train_rssm: True
    train_legality: True
    train_controller: True
  data_collection: # Data collection settings
    num_episodes: 1000
    final_reward_mode: 0 # 0: 10 for win and -10 for loss, 1: 100 for win and -100 for loss, else: 1 for win and -1 for loss
    incremental_reward_mode: 1 # 0: no incremental reward, 1: incremental reward with potential calculation, 2: incremental reward with LOSING penalty
    class1: all
    class2: all
    deck1: all
    deck2: all
    sampling_agent: RandomAgent
    score_method: None
  world_model: # General World model settings
    num_cats: 16
    cat_dim: 16
  encoder: # MultiHeadAutoEncoder settings
    epochs: 20
    batch_size: 128
    learning_rate: 0.001
    cont_hidden_dim: 128
    disc_hidden_dim: 32
  rssm: # RNN settings
    epochs: 100
    batch_size: 24
    hidden_dim: 256
    learning_rate: 0.00005
    sequence_length: 32
    print_every: 1
    predict_reward: True
  legality_net:
    widths: [512, 512]
    epochs: 10
    batch_size: 128
    learning_rate: 0.0003
  controller: # Controller settings (PPO-based)
    model_type: Mask
    use_dreamer_callback: False
    policy_layers: [1024, 512, 256]
    value_layers: [512, 512]
    learning_rate: 0.00001
    gamma: 0.99
    gae_lambda: 0.95
    max_steps: 256
    n_steps: 512
    batch_size: 256
    clip_range: 0.2
    ent_coef: 0.0
    n_epochs: 5
    seed: 42
    total_steps: 1_000_000
    eval_episodes: 5
    player_class: all
    player_deck: all
    opponent_class: all
    opponent_deck: all
    mirror_matches: False
    opponent_agent: RandomAgent
    opponent_method: None
    ema_sigma: 0.98
    return_percentile: 95.0
    osfp_alpha: None
    osfp_update_freq: None
    tensorboard_log: TENSORBOARD_FOLDER
```

</details>


<details>
<summary>Click to expand the parameter explanation</summary>

The configuration file is divided into several sections, each containing different parameters: `general`, `data_collection`, `world_model`, `encoder`, `rssm`, `legality_net` and `controller`. Each section contains parameters that control the behavior of the agent. The most important parameters are described below:
- `general`: General settings for the agent. 
    - `name`: The `name` parameter specifies the name of the agent. 
    - `device`: The `device` parameter specifies the device to use for training (e.g., `cuda` or `cpu`). 
    - `embedded`: The `embedded` parameter specifies whether to use an embedded model or not. 
    - `deck_include`/`deck_include_v2`: The `deck_include` and `deck_include_v2` parameters specify whether to include the deck in the state representation. 
    - `collect_data`: The `collect_data` parameter specifies whether to collect data during training. 
    - `collect_masks`: The `collect_masks` parameter specifies whether to collect masks during training. 
    - `train_*`: The `train_encoder`, `train_rssm`, `train_legality`, and `train_controller` parameters specify whether to train the encoder, RSSM, legality net, and controller, respectively.
- `data_collection`:
    - `num_episodes`: The `num_episodes` parameter specifies the number of episodes to collect data for. 
    - `final_reward_mode`: The `final_reward_mode` parameter specifies the reward mode for the final reward. 
    - `incremental_reward_mode`: The `incremental_reward_mode` parameter specifies the reward mode for the incremental reward. 
    - `class1`/`class2`: The `class1` and `class2` parameters specify the classes to use for data collection. 
    - `deck1`/`deck2`: The `deck1` and `deck2` parameters specify the decks to use for data collection. 
    - `sampling_agent`: The `sampling_agent` parameter specifies the agent to use for sampling. 
    - `score_method`: The `score_method` parameter specifies the method to use for scoring.
- `world_model`: The `world_model` section contains parameters for the world model.
    - `num_cats`: The `num_cats` parameter specifies the number of categories for the world model. 
    - `cat_dim`: The `cat_dim` parameter specifies the dimension of the categories.
- `encoder`: The `encoder` section contains parameters for the encoder.
    - `epochs`: The `epochs` parameter specifies the number of epochs to train the encoder. 
    - `batch_size`: The `batch_size` parameter specifies the batch size for training the encoder. 
    - `learning_rate`: The `learning_rate` parameter specifies the learning rate for training the encoder. 
    - `cont_hidden_dim`: The `cont_hidden_dim` parameter specifies the hidden dimension for the continuous part of the encoder. 
    - `disc_hidden_dim`: The `disc_hidden_dim` parameter specifies the hidden dimension for the discrete part of the encoder.
- `rssm`: The `rssm` section contains parameters for the RSSM.
    - `epochs`: The `epochs` parameter specifies the number of epochs to train the RSSM. 
    - `batch_size`: The `batch_size` parameter specifies the batch size for training the RSSM. 
    - `hidden_dim`: The `hidden_dim` parameter specifies the hidden dimension for the RSSM. 
    - `learning_rate`: The `learning_rate` parameter specifies the learning rate for training the RSSM. 
    - `sequence_length`: The `sequence_length` parameter specifies the sequence length for training the RSSM. 
    - `print_every`: The `print_every` parameter specifies how often to print the training progress. 
    - `predict_reward`: The `predict_reward` parameter specifies whether to predict the reward or not.
- `legality_net`: The `legality_net` section contains parameters for the legality net.
    - `widths`: The `widths` parameter specifies the widths of the layers in the legality net. 
    - `epochs`: The `epochs` parameter specifies the number of epochs to train the legality net. 
    - `batch_size`: The `batch_size` parameter specifies the batch size for training the legality net. 
    - `learning_rate`: The `learning_rate` parameter specifies the learning rate for training the legality net.
- `controller`: The `controller` section contains parameters for the controller.
    - `model_type`: The `model_type` parameter specifies the type of model to use for the controller. (Options: `Mask`, `MaskRNN`, `RNN`, `PPO`)
    - `use_dreamer_callback`: The `use_dreamer_callback` parameter specifies whether to use the Dreamer callback or not. 
    - `policy_layers`: The `policy_layers` parameter specifies the layers for the policy network. 
    - `value_layers`: The `value_layers` parameter specifies the layers for the value network. 
    - `learning_rate`: The `learning_rate` parameter specifies the learning rate for training the controller. 
    - `gamma`: The `gamma` parameter specifies the discount factor for training the controller. 
    - `gae_lambda`: The `gae_lambda` parameter specifies the GAE lambda for training the controller. 
    - `max_steps`: The `max_steps` parameter specifies the maximum number of steps for training the controller. 
    - `n_steps`: The `n_steps` parameter specifies the number of steps for training the controller. 
    - `batch_size`: The `batch_size` parameter specifies the batch size for training the controller. 
    - `clip_range`: The `clip_range` parameter specifies the clip range for training the controller. 
    - `ent_coef`: The `ent_coef` parameter specifies the entropy coefficient for training the controller. 
    - `n_epochs`: The `n_epochs` parameter specifies the number of epochs for training the controller. 
    - `seed`: The `seed` parameter specifies the seed for random number generation. 
    - `total_steps`: The `total_steps` parameter specifies the total number of steps for training the controller. 
    - `eval_episodes`: The `eval_episodes` parameter specifies the number of episodes to evaluate during training. 
    - `player_class`/`player_deck`: The `player_class` and `player_deck` parameters specify the class and deck to use for the player. (See the list of classes and decks in the `src/run_game.py` file or general README.md)
    - `opponent_class`/`opponent_deck`: The `opponent_class` and `opponent_deck` parameters specify the class and deck to use for the opponent. (See the list of classes and decks in the `src/run_game.py` file or general README.md)
    - `mirror_matches`: The `mirror_matches` parameter specifies whether to use mirror matches or not.
    - `opponent_agent`: The opponent agent to use during training. (See the list of agents in the `src/run_game.py` file or general README.md)
    - `opponent_method`: The opponent method to use during training. (Only used for `GreedyAgent`, options: `aggro`, `control`, `ramp`)
    - `ema_sigma`: The EMA sigma for training the controller. (Used by the `Dreamer` callback)
    - `return_percentile`: The return percentile for training the controller. (Used by the `Dreamer` callback)
    - `osfp_alpha`: The OSFP alpha for training the controller. (Only used when having the `OSFPAgent` as opponent)
    - `osfp_update_freq`: The OSFP update frequency for training the controller. (Only used when having the `OSFPAgent` as opponent)
    - `tensorboard_log`: The tensorboard log folder for training the controller.

</details>

## Training
To train the agent, run the following command:
```bash
python src/train_world_model.py 
```

It automatically detects the new model in the `world_model_settings.yaml` file and starts training it. The training process will save the model checkpoints and logs in the specified folders.