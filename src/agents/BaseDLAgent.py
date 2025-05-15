from agents.HearthstoneAgent import HearthstoneAgent
from env.hearthstone.HearthGym import HearthstoneEnv
import numpy as np
from numba import njit

INT_MAX = np.iinfo(np.int32).max
INT_MIN = np.iinfo(np.int32).min

# Global scaling array (make sure it’s a NumPy array so that Numba can work with it)
SCALING = np.array([21.5, 33.6, 41.1, 19.4, 54.0, 60.5, 88.5, 84.7])

@njit
def score_custom_num(
    hero_hp, op_hero_hp,
    own_count, opp_count,
    own_health, own_atk,
    opp_health, opp_atk,
    scaling
    ):
    
    # Lethal checks
    if op_hero_hp < 1:
        return INT_MAX
    if hero_hp < 1:
        return INT_MIN

    score = (
        scaling[0] * hero_hp -
        scaling[1] * op_hero_hp +
        scaling[2] * own_count -
        scaling[3] * opp_count +
        scaling[4] * own_health +
        scaling[5] * own_atk -
        scaling[6] * opp_health -
        scaling[7] * opp_atk
        )
    
    return int(round(score))



class BaseDynamicLookaheadAgent(HearthstoneAgent):
    def __init__(self):
        super().__init__()

    def act(self, observation, valid_actions=None, action_mask=None, env: HearthstoneEnv = None):
        player = env.game.current_player

        valid_sims = self.simulate(action_list=valid_actions, env=env)

        optcount = len(valid_sims)
        
        if optcount >= 25:
            depth = 1
        elif optcount >= 5:
            depth = 2
        else:
            depth = 3

        best_action = None
        best_score = INT_MIN

        for (action, sim_env) in valid_sims:
            candidate_score = self._recursive_score(sim_env, player.name, depth)
            if candidate_score > best_score:
                best_action = action
                best_score = candidate_score

        return best_action

    def _recursive_score(self, env_state, player_name, depth):
        """
        Recursively compute the max score if it's still our turn.
        Compare against direct state score as well.
        """
        max_val = INT_MIN

        # If depth > 0 and it's still the same player's turn => try further moves
        if depth > 0 and env_state.game.current_player.name == player_name:
            next_actions, _ = env_state.get_valid_actions()
            valid_next = self.simulate(next_actions, env_state)
            for (new_actions, new_environment) in valid_next:
                score_cand = self._recursive_score(new_environment, player_name, depth - 1)
                if score_cand > max_val:
                    max_val = score_cand

        # Get immediate state's score
        direct_score = self.score_state(env_state, player_name)
        if direct_score > max_val:
            max_val = direct_score

        return max_val

    def score_state(self, env_state, player_name):
        """
        Score using our custom jitted scoring function.
        """
        current_player = env_state.game.current_player
        if current_player.name == player_name:
            controller = current_player
        else:
            controller = current_player.opponent

        # Extract numeric features from the controller
        hero_hp = controller.hero.health
        op_hero_hp = controller.opponent.hero.health

        own_board = controller.field
        opp_board = controller.opponent.field

        own_count = len(own_board)
        opp_count = len(opp_board)

        own_health = 0
        own_atk = 0
        for m in own_board:
            own_health += m.health
            own_atk += m.atk

        opp_health = 0
        opp_atk = 0
        for m in opp_board:
            opp_health += m.health
            opp_atk += m.atk

        # Call the njitted function
        return score_custom_num(
            hero_hp, op_hero_hp,
            own_count, opp_count,
            own_health, own_atk,
            opp_health, opp_atk,
            SCALING
            )