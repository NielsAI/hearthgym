# Readme for running experiments with HearthGym

## Setup
The main component to configure when running experiments is the `src/experiments/experiments_config.yaml` file. This file can have multiple key entries, where each is treated as a separate experiment. Below are the required parameters for creating a new experiment and their explanations.


<details>
<summary>Click to expand the YAML code</summary>

```yaml
experiments:
  EXAMPLE_EXPERIMENT:
    enabled: True
    games: 1000
    agent1_index: 6
    agent2_index: 0
    ppo_model1: PATH_TO_PPO_MODEL
    ppo_model2: none
    ppo_type1: PPO_TYPE_P1 (e.g. Mask, MaskRNN)
    ppo_type2: PPO_TYPE_P2 (e.g. Mask, MaskRNN)
    score_method1: SCORE_METHOD_P1 (e.g. None, aggro, control)
    score_method2: SCORE_METHOD_P2 (e.g. None, aggro, control)
    class1: CLASS_ID_P1 (e.g. 'all', 2, 3, 4)
    class2: CLASS_ID_P2 (e.g. 'all', 2, 3, 4)
    deck1: DECK_ID_P1 (e.g. 'all', 0, 1)
    deck2: DECK_ID_P2 (e.g. 'all', 0, 1)
    embedded: False
    deck_include: False
    deck_include_v2: False
    seed: 42
    folder: LOG_FOLDER_NAME
    save_observations: False
    encoder1: ENCODER_PATH_P1
    encoder2: ENCODER_PATH_P2
    rssm1: RSSM_PATH_P1
    rssm2: RSSM_PATH_P2
    mirror: False
```

</details>


<details>
<summary>Click to expand the parameter explanation</summary>

The configuration file is divided into several sections, each containing different parameters: `general`, `data_collection`, `encoder`, and `controller`. Each section contains parameters that control the behavior of the agent. The most important parameters are described below:
- `enabled`: Whether the experiment is enabled or not. If set to `False`, the experiment will be skipped.
- `games`: The number of games to play in the experiment.
- `agent1_index` and `agent2_index`: The indices of the agents that will play the game. The agents are defined in the `agents` folder.
- `ppo_model1` and `ppo_model2`: The paths to the PPO models for agent 1 and agent 2, respectively. If set to `none`, the agent will use a random agent.
- `ppo_type1` and `ppo_type2`: The types of PPO models for agent 1 and agent 2, respectively. Options are `Mask`, `RNN`, `MaskRNN`, or `PPO`.
- `score_method1` and `score_method2`: The scoring methods for agent 1 and agent 2, respectively. Options are `None`, `aggro`, or `control`. (Only required for `GreedyAgent`)
- `class1` and `class2`: The ids of the classes of the agents. The classes are defined in the `fireplace` package and the options are listed below.
- `deck1` and `deck2`: The ids of the decks of the agents. The decks are defined in the `fireplace` package and the options are listed below.
- `embedded`: Whether to use the embedded version of the model.
- `deck_include`/`deck_include_v2`: Whether to include the deck in the observation.
- `seed`: The seed for the random number generator.
- `folder`: The folder where the logs will be saved.
- `save_observations`: Whether to save the observations during the experiment.
- `encoder1` and `encoder2`: The paths to the encoders for agent 1 and agent 2, respectively. (Only required for `EncodedPPO` agents and `WorldModel` agents)
- `rssm1` and `rssm2`: The paths to the RSSM models for agent 1 and agent 2, respectively. (Only required for `WorldModel` agents)
- `mirror`: Whether to play mirror matches or not. If set to `True`, the same agent will play against itself.
    

</details>

## Running the Experiments
To run the experiments, you need to run the `src/run_experiments.py` script:
```bash
python src/run_experiments.py
```