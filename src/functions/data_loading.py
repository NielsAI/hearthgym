import pandas as pd
import numpy as np

def generate_feature_names(embedded=False, deck_include=False, deck_include_v2=False, 
                           embedding_size=16):
    """
    Generate feature names along with indices for continuous and categorical features.
    
    :param embedded: Boolean indicating if embedded features are included.
    :param deck_include: Boolean indicating if deck inclusion features are included.
    :param deck_include_v2: Boolean indicating if deck inclusion v2 features are included.
    :param embedding_size: Size of the embedding for deck inclusion v2.
    :return: Tuple of (features, cont_indices, cat_indices)
    """
    N_MINION_STATS = 16
    N_CARD_CLASSES = 15
    MAX_BOARD_SIZE=7
    
    features = []
    cont_indices = []
    cat_indices = []
    idx = 0  # index counter

    # 1. Player Class: one-hot vector with N_CARD_CLASSES entries. (Categorical)
    for i in range(N_CARD_CLASSES):
        features.append(f"player_class_{i}")
        cat_indices.append(idx)
        idx += 1

    # 2. Basic Player Info.
    # Continuous: player_mana, max_mana, hero_health, hero_attack, hero_armor
    for name in ["player_mana", "max_mana", "hero_health", "hero_attack", "hero_armor"]:
        features.append(name)
        cont_indices.append(idx)
        idx += 1
    # Categorical: has_choice (binary)
    features.append("has_choice")
    cat_indices.append(idx)
    idx += 1

    # 3. Opponent Hero Health (continuous)
    features.append("opponent_health")
    cont_indices.append(idx)
    idx += 1

    # 4. Player Board Stats for each board slot.
    # For each slot, we add N_MINION_STATS features.
    # We treat indices 0,1,2 ("atk", "health", "max_health") as continuous; the rest as categorical.
    minion_stat_names = ["atk", "health", "max_health", "can_attack", 
                         "divine_shield", "stealthed", "frozen", "silenced",
                         "windfury", "mega_windfury", "immune_while_attacking",
                         "has_inspire", "has_battlecry", "has_deathrattle",
                         "has_overkill", "lifesteal"]
    for i in range(MAX_BOARD_SIZE):
        for j, stat in enumerate(minion_stat_names):
            features.append(f"board_minion_{i}_{stat}")
            if j < 3:
                cont_indices.append(idx)
            else:
                cat_indices.append(idx)
            idx += 1

    # 5. Opponent Board Stats.
    for i in range(MAX_BOARD_SIZE):
        for j, stat in enumerate(minion_stat_names):
            features.append(f"opp_board_minion_{i}_{stat}")
            if j < 3:
                cont_indices.append(idx)
            else:
                cat_indices.append(idx)
            idx += 1

    # 6. Additional continuous game context features:
    for name in ["opponent_mana", "opponent_max_mana", "opponent_hand_size", "turn_number",
                 "hero_weapon_attack", "hero_weapon_durability"]:
        features.append(name)
        cont_indices.append(idx)
        idx += 1
    # 'hero_power_available' is a binary flag -> categorical.
    features.append("hero_power_available")
    cat_indices.append(idx)
    idx += 1

    # 7. Optional: Deck Inclusion (continuous counts)
    if deck_include:
        collection_size = 1016  # assumed fixed collection size
        for i in range(collection_size):
            features.append(f"deck_count_{i}")
            cont_indices.append(idx)
            idx += 1

    # 8. Optional: Deck Inclusion v2 (categorical IDs)
    if deck_include_v2:
        for j in range(embedding_size):
            features.append(f"embedding_deck_{j}")
            cont_indices.append(idx)
            idx += 1

    # 9. Optional: Embedded Card Features
    if embedded:
        # For hand: we produce one vector of length embedding_size.
        for j in range(embedding_size):
            features.append(f"embedding_hand_{j}")
            cont_indices.append(idx)
            idx += 1
        # For player's board:
        for j in range(embedding_size):
            features.append(f"embedding_board_{j}")
            cont_indices.append(idx)
            idx += 1
        # For opponent's board:
        for j in range(embedding_size):
            features.append(f"embedding_opp_board_{j}")
            cont_indices.append(idx)
            idx += 1

    return features, cont_indices, cat_indices

def load_norm_params_with_type(csv_file):
    """
    Load normalization parameters from a CSV file that includes a 'type' column.
    :param csv_file: Path to the CSV file containing normalization parameters.
    :return: A tuple containing:
        - norm_dict: Dictionary with normalization parameters for each feature.
        - cont_features: List of continuous features.
        - cat_features: List of categorical features.
    """
    df = pd.read_csv(csv_file)
    norm_dict = {}
    cont_features = []
    cat_features = []
    
    for idx, row in df.iterrows():
        feature = row['feature']
        norm_dict[feature] = {
            'mean': float(row['mean']),
            'std': float(row['std']),
            'max': float(row['max']),
            'min': float(row['min']),
            'type': row['type']
        }
        if row['type'] == 'continuous':
            cont_features.append(feature)
        elif row['type'] == 'categorical':
            cat_features.append(feature)
            
    return norm_dict, cont_features, cat_features

def align_norm_params(feature_names, norm_dict):
    """
    Given an ordered list of feature names (from your observation vector) and
    a normalization dictionary (with 'type' info), create aligned arrays for
    continuous features only.
    For features not found in norm_dict (or non-continuous ones), defaults are used.
    :param feature_names: List of feature names in the order they appear in the observation vector.
    :param norm_dict: Dictionary containing normalization parameters for each feature.
    :return: Tuple of aligned means and standard deviations for continuous features.
    """
    # Set mode to ignore categorical features
    MODE = 2
    
    aligned_means = np.zeros(len(feature_names), dtype=np.float32)
    aligned_stds  = np.ones(len(feature_names), dtype=np.float32)

    for i, fname in enumerate(feature_names):
        if MODE == 1:
            aligned_means[i] = norm_dict[fname]['mean']
            aligned_stds[i] = norm_dict[fname]['std'] if norm_dict[fname]['std'] != 0 else 1.0
        elif MODE == 2:
            if fname in norm_dict and norm_dict[fname]['type'] == 'continuous':
                aligned_means[i] = norm_dict[fname]['mean']
                aligned_stds[i] = norm_dict[fname]['std'] if norm_dict[fname]['std'] != 0 else 1.0
            else:
                # For categorical features or if not found, leave defaults.
                aligned_means[i] = 0.0
                aligned_stds[i] = 1.0
    return aligned_means, aligned_stds