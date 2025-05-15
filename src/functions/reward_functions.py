from numba import njit

@njit
def calculate_potential_numba(hero_healths, minions, mana, hand_size, hand_weight, minion_weight):
    """
    Compute a simple potential function for a given state.
    
    :param hero_healths: Tuple of hero healths (your hero, opponent's hero).
    :param minions: Tuple of minion counts (your minions, opponent's minions).
    :param mana: Tuple of mana (your mana, opponent's mana).
    :param hand_size: Size of the hand.
    :param hand_weight: Weight for the hand size in potential calculation.
    :param minion_weight: Weight for the minion count in potential
    :return: A float representing the potential of the state.
    """
    potential = (hero_healths[0] + mana[0] + hand_size * hand_weight + minions[0] * minion_weight) - \
                (hero_healths[1] + mana[1] + minions[1] * minion_weight)
    return potential

    
@njit
def calculate_intermediate_reward_numba(
    intermediate_reward_mode,
    prev_hero_healths, current_hero_healths,
    prev_minions, current_minions,
    prev_mana, current_mana,
    prev_hand_size, current_hand_size,
    gamma = 0.99, hand_weight = 1.0, minion_weight = 1.0, potential_weight = 0.1, # Initial values
    ) -> float:
    """
    Compute the intermediate reward including the original reward components and a potential-based shaping term.
    
    :param intermediate_reward_mode: Mode for calculating the reward.
    :param prev_hero_healths: Previous hero healths (your hero, opponent's hero).
    :param current_hero_healths: Current hero healths (your hero, opponent's hero).
    :param prev_minions: Previous minion counts (your minions, opponent's minions).
    :param current_minions: Current minion counts (your minions, opponent's minions).
    :param prev_mana: Previous mana (your mana, opponent's mana).
    :param current_mana: Current mana (your mana, opponent's mana).
    :param prev_hand_size: Previous hand size.
    :param current_hand_size: Current hand size.
    :param gamma: Discount factor for potential-based shaping.
    :param hand_weight: Weight for the hand size in potential calculation.
    :param minion_weight: Weight for the minion count in potential calculation.
    :param potential_weight: Weight for the potential-based shaping term.
    :return: A float representing the computed reward.
    """
    reward = 0.0

    if intermediate_reward_mode == 2:
        # A. Damage dealt to opponent’s hero
        damage_dealt = prev_hero_healths[1] - current_hero_healths[1]
        if damage_dealt > 0:
            reward += damage_dealt * 0.05

        # B. Damage taken by agent’s hero
        damage_taken = prev_hero_healths[0] - current_hero_healths[0]
        if damage_taken > 0:
            reward -= damage_taken * 0.05

        # C. Opponent minions killed
        minions_killed = prev_minions[1] - current_minions[1]
        if minions_killed > 0:
            reward += minions_killed * 0.3

        # D. Agent minions added
        minions_added = current_minions[0] - prev_minions[0]
        if minions_added > 0:
            reward += minions_added * 0.2

        # E. Mana used (assuming mana decreases when used)
        mana_used = prev_mana[0] - current_mana[0]
        if mana_used > 0:
            reward += mana_used * 0.1

        # F. Cards played (a fixed bonus per card played)
        cards_played = prev_hand_size - current_hand_size
        if cards_played > 0:
            reward += 0.1

    # G. Potential-based reward shaping:
    potential_prev = calculate_potential_numba(
        prev_hero_healths, prev_minions, 
        prev_mana, prev_hand_size,
        hand_weight, minion_weight
        )
    potential_curr = calculate_potential_numba(
        current_hero_healths, current_minions, 
        current_mana, current_hand_size,
        hand_weight, minion_weight
        )
    
    # Calculate the potential-based shaping reward
    shaping_reward = gamma * potential_curr - potential_prev
    reward += potential_weight * shaping_reward

    # Clip reward to the range [-1, 1]
    if reward > 1.0:
        reward = 1.0
    elif reward < -1.0:
        reward = -1.0

    return reward