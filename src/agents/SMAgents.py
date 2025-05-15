import numpy as np
from numba import njit
from agents.HearthstoneAgent import HearthstoneAgent
from env.hearthstone.HearthGym import HearthstoneEnv

# Define constants for lethal outcomes.
INT_MAX = np.iinfo(np.int32).max
INT_MIN = np.iinfo(np.int32).min


@njit
def better_score_num(
    hero_hp, opp_hp, board_count, op_board_count,
    own_minion_health, own_minion_atk,
    opp_minion_health, opp_minion_atk
    ):
    # Handle lethal conditions.
    if opp_hp < 1:
        return INT_MAX
    if hero_hp < 1:
        return INT_MIN

    score = 2 * hero_hp - 3 * opp_hp
    score += 4 * board_count - 6 * op_board_count
    score += 5 * own_minion_health + 6 * own_minion_atk
    score -= 9 * opp_minion_health + 8 * opp_minion_atk
    return score

class NaiveBetterScore:
    def __init__(self, controller):
        self.controller = controller

    def rate(self):
        if self.controller is None:
            return 0

        hero_hp = self.controller.hero.health
        opp_hp = self.controller.opponent.hero.health

        # Lethal checks are done in the jitted function.
        board = self.controller.field
        op_board = self.controller.opponent.field

        board_count = len(board)
        op_board_count = len(op_board)

        own_minion_health = 0
        own_minion_atk = 0
        for m in board:
            own_minion_health += m.health
            own_minion_atk += m.atk

        opp_minion_health = 0
        opp_minion_atk = 0
        for m in op_board:
            opp_minion_health += m.health
            opp_minion_atk += m.atk

        return better_score_num(hero_hp, opp_hp, board_count, op_board_count,
                                own_minion_health, own_minion_atk,
                                opp_minion_health, opp_minion_atk)

    def mulligan_rule(self, hand_cards):
        return [c.id for c in hand_cards if c.cost > 3]

class NaiveScoreLookaheadAgent(HearthstoneAgent):
    def __init__(self):
        super().__init__()

    def act(self, observation, valid_actions=None, action_mask=None, env: HearthstoneEnv = None):
        player = env.game.current_player
        valid_opts = self.simulate(action_list=valid_actions, env=env)
        optcount = len(valid_opts)

        if valid_opts:
            scored_actions = [self.score_action(opt, st, player.name, optcount) for opt, st in valid_opts]
            best_action = max(scored_actions, key=lambda x: x[1])[0]
            valid_opts = None  # free memory
            return best_action
        else:
            end_turn = [o for o in valid_actions if o[0] == 0]
            return end_turn[0] if end_turn else None

    def score_action(self, action, state_env, player_name, optcount):
        depth = 3
        if optcount >= 5:
            depth = 2
        if optcount >= 25:
            depth = 1

        final_score = self.recursive_score(state_env, player_name, depth)
        return action, final_score

    def recursive_score(self, env_state, player_name, depth):
        max_score = -np.inf
        if depth > 0 and env_state.game.current_player.name == player_name:
            next_opts_list, _ = env_state.get_valid_actions()
            next_opts = self.simulate(action_list=next_opts_list, env=env_state)
            for sub_o, sub_s in next_opts:
                candidate = self.recursive_score(sub_s, player_name, depth - 1)
                if candidate > max_score:
                    max_score = candidate
            next_opts = None
        max_score = max(max_score, self.score_state(env_state, player_name))
        return max_score

    def score_state(self, env_state, player_name):
        controller = (env_state.game.current_player if env_state.game.current_player.name == player_name 
                      else env_state.game.current_player.opponent)
        return NaiveBetterScore(controller).rate()

class WeightedBetterScore:
    def __init__(self, controller):
        self.controller = controller

    def rate(self):
        if self.controller is None:
            return 0

        hero_hp = self.controller.hero.health
        opp_hp = self.controller.opponent.hero.health

        board = self.controller.field
        op_board = self.controller.opponent.field

        board_count = len(board)
        op_board_count = len(op_board)

        own_minion_health = 0
        own_minion_atk = 0
        for m in board:
            own_minion_health += m.health
            own_minion_atk += m.atk

        opp_minion_health = 0
        opp_minion_atk = 0
        for m in op_board:
            opp_minion_health += m.health
            opp_minion_atk += m.atk

        return better_score_num(hero_hp, opp_hp, board_count, op_board_count,
                                own_minion_health, own_minion_atk,
                                opp_minion_health, opp_minion_atk)

    def mulligan_rule(self, hand_cards):
        return [c.id for c in hand_cards if c.cost > 3]

class WeightedScoreAgent(HearthstoneAgent):
    def __init__(self):
        super().__init__()

    def act(self, observation, valid_actions=None, action_mask=None, env: HearthstoneEnv = None):
        player = env.game.current_player
        valid_opts = self.simulate(action_list=valid_actions, env=env)
        optcount = len(valid_opts)

        if valid_opts:
            scored_actions = [self.score_action(opt, st, player.name, optcount) for opt, st in valid_opts]
            best_action = max(scored_actions, key=lambda x: x[1])[0]
            valid_opts = None
            return best_action
        else:
            end_turn = [o for o in valid_actions if o[0] == 0]
            return end_turn[0] if end_turn else None

    def score_action(self, action, state_env, player_name, optcount):
        depth = 3
        if optcount >= 5:
            depth = 2
        if optcount >= 25:
            depth = 1

        final_score = self.recursive_score(state_env, player_name, depth)
        return action, final_score

    def recursive_score(self, env_state, player_name, depth):
        max_score = -np.inf
        if depth > 0 and env_state.game.current_player.name == player_name:
            next_opts_list, _ = env_state.get_valid_actions()
            next_opts = self.simulate(action_list=next_opts_list, env=env_state)
            for sub_o, sub_s in next_opts:
                candidate = self.recursive_score(sub_s, player_name, depth - 1)
                if candidate > max_score:
                    max_score = candidate
            next_opts = None
        max_score = max(max_score, self.score_state(env_state, player_name))
        return max_score

    def score_state(self, env_state, player_name):
        controller = (env_state.game.current_player if env_state.game.current_player.name == player_name 
                      else env_state.game.current_player.opponent)
        return WeightedBetterScore(controller).rate()
