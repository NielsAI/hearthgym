# README for creating PPO agents

This repository contains a framework for creating PPO agents for the Hearthstone game. The framework is based on the `fireplace` package, which is a Python implementation of the Hearthstone game. The framework is designed to be used with reinforcement learning algorithms to train agents to play Hearthstone.

## Setup
The main component to configure when creating a new agent is the `src/models/PPO/ppo_model_settings.yaml` file. This file has multiple key entries, the first being `template`, showing all required parameters for creating a new agent. The new agent configuration should be stated with the key `new_model`, this is then used by the script. Other entries can be stored in this file to keep track of previously created models. Below are the required parameters for creating a new agent and their explanations.


<details>
<summary>Click to expand the YAML code</summary>

```yaml
new_model:
  name: MODEL_NAME
  model_type: Mask # Model types: Mask | RNN | MaskRNN | PPO
  device: cuda
  embedded: False
  deck_include: False
  deck_include_v2: False
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

The configuration file contains the following parameters:
- `name`: The name of the model. This will be used to create the folder where the model will be saved.
- `model_type`: The type of model to use. Options are `Mask`, `RNN`, `MaskRNN`, or `PPO`.
- `device`: The device to use for training. Options are `cuda` or `cpu`.
- `embedded`: Whether to use the embedded version of the model.
- `deck_include`: Whether to include the deck in the observation. 
- `deck_include_v2`: Whether to include the deck in the observation (v2). 
- `policy_layers`: The layers of the policy network.
- `value_layers`: The layers of the value network. 
- `learning_rate`: The learning rate for the optimizer. 
- `gamma`: The discount factor for the reward.
- `gae_lambda`: The lambda parameter for the Generalized Advantage Estimation.
- `n_steps`: The number of steps to run for each environment per update.
- `batch_size`: The batch size for training.
- `clip_range`: The clipping range for the PPO algorithm.
- `ent_coef`: The coefficient for the entropy term.
- `n_epochs`: The number of epochs to train the model.
- `seed`: The random seed for the model.
- `total_steps`: The total number of steps to train the model.
- `eval_episodes`: The number of episodes to evaluate the model.
- `final_reward_mode`: The mode for the final reward. Options are:
    - `0`: 10 for win and -10 for loss
    - `1`: 100 for win and -100 for loss
    - else: 1 for win and -1 for loss
- `incremental_reward_mode`: The mode for the incremental reward. Options are:
    - `0`: No incremental reward
    - `1`: Incremental reward based potential calculation
    - `2`: Incremental reward based potential calculation and advantage function
- `player_class`/`player_deck`: The `player_class` and `player_deck` parameters specify the class and deck to use for the player. (See the list of classes and decks in the `src/run_game.py` file or general README.md)
- `opponent_class`/`opponent_deck`: The `opponent_class` and `opponent_deck` parameters specify the class and deck to use for the opponent. (See the list of classes and decks in the `src/run_game.py` file or general README.md)
- `mirror_matches`: Whether to allow mirror matches.
- `opponent_agent`: The agent to use for the opponent.
- `opponent_method`: The method to use for the opponent.
- `osfp_alpha`: The alpha parameter for the OSFP algorithm.
- `osfp_update_freq`: The update frequency for the OSFP algorithm. 
- `tensorboard_log`: The folder where the TensorBoard logs will be saved. 
    

</details>

## Training
To train the agent, run the following command:
```bash
python src/train_ppo_model.py
```

It automatically detects the new model in the `ppo_model_settings.yaml` file and starts training it. The training process will save the model checkpoints and logs in the specified folders.