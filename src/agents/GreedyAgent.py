import numpy as np
from agents.HearthstoneAgent import HearthstoneAgent
from env.hearthstone.HearthGym import HearthstoneEnv

from numba import njit

INT_MAX = 2147483647
INT_MIN = -2147483648

@njit
def score_aggro_num(hero_hp, op_hero_hp, board_count, op_board_count, minion_tot_atk, op_minion_tot_health_taunt):
    if op_hero_hp < 1:
        return INT_MAX
    if hero_hp < 1:
        return INT_MIN

    result = 0
    # Bonus if opponent board is empty but we have minions
    if op_board_count == 0 and board_count > 0:
        result += 1000
    # Penalize if opponent has taunt minions
    if op_minion_tot_health_taunt > 0:
        result += op_minion_tot_health_taunt * -1000
    result += minion_tot_atk
    result += (hero_hp - op_hero_hp) * 1000
    return result

@njit
def score_control_num(hero_hp, op_hero_hp, board_count, op_board_count, minion_tot_atk, minion_tot_health_taunt, op_minion_tot_health_taunt):
    if op_hero_hp < 1:
        return INT_MAX
    if hero_hp < 1:
        return INT_MIN

    result = 0
    if op_board_count == 0 and board_count > 0:
        result += 1000
    result += (board_count - op_board_count) * 50
    result += (minion_tot_health_taunt - op_minion_tot_health_taunt) * 25
    result += minion_tot_atk
    result += (hero_hp - op_hero_hp) * 10
    return result

@njit
def score_ramp_num(hero_hp, op_hero_hp, board_count, op_board_count,
                   minion_tot_atk, minion_tot_health, minion_tot_health_taunt,
                   op_minion_tot_atk, op_minion_tot_health, op_minion_tot_health_taunt):
    if op_hero_hp < 1:
        return INT_MAX
    if hero_hp < 1:
        return INT_MIN

    result = 0
    if op_board_count == 0 and board_count > 0:
        result += 5000
    result += (board_count - op_board_count) * 50
    if op_minion_tot_health_taunt > 0:
        result += minion_tot_health_taunt * -500
    result += minion_tot_atk
    result += (hero_hp - op_hero_hp) * 10
    result += (minion_tot_health - op_minion_tot_health) * 10
    result += (minion_tot_atk - op_minion_tot_atk) * 20
    return result

class GreedyAgent(HearthstoneAgent):
    def __init__(self, score_method: str = "aggro"):
        super().__init__()
        
        self.score_method = score_method

    def act(self, observation, valid_actions=None, action_mask=None, env: HearthstoneEnv=None):
        """
        Decide on an action. If in mulligan, do that; otherwise pick
        the action that has the best final score from the list of valid actions.
        """
        player = env.game.current_player

        valid_sims = self.simulate(action_list=valid_actions, env=env)

        best_action = None
        best_score = -np.inf

        for (action, sim_env) in valid_sims:
            s = self.score_state(sim_env, player.name)
            if s > best_score:
                best_action = action
                best_score = s
                
        # Remove the valid sims to free up memory
        valid_sims = None

        return best_action
    
    def score_state(self, env_state, player_name):
        # Identify the "controller" in this state 
        if env_state.game.current_player.name == player_name:
            controller = env_state.game.current_player
        else:
            controller = env_state.game.current_player.opponent

        # Extract the numeric values needed for scoring.
        hero_hp = controller.hero.health
        op_hero_hp = controller.opponent.hero.health
        board_count = len(controller.field)
        op_board_count = len(controller.opponent.field)
        minion_tot_atk = sum(m.atk for m in controller.field)
        minion_tot_health = sum(m.health for m in controller.field)
        minion_tot_health_taunt = sum(m.health for m in controller.field if m.taunt)
        op_minion_tot_atk = sum(m.atk for m in controller.opponent.field)
        op_minion_tot_health = sum(m.health for m in controller.opponent.field)
        op_minion_tot_health_taunt = sum(m.health for m in controller.opponent.field if m.taunt)

        if self.score_method == "aggro":
            return score_aggro_num(
                hero_hp, op_hero_hp,
                board_count, op_board_count,
                minion_tot_atk, op_minion_tot_health_taunt
                )
        elif self.score_method == "control":
            return score_control_num(
                hero_hp, op_hero_hp,
                board_count, op_board_count,
                minion_tot_atk, minion_tot_health_taunt,
                op_minion_tot_health_taunt
                )
        else:
            return score_ramp_num(
                hero_hp, op_hero_hp,
                board_count, op_board_count,
                minion_tot_atk, minion_tot_health,
                minion_tot_health_taunt,
                op_minion_tot_atk, op_minion_tot_health,
                op_minion_tot_health_taunt
                )