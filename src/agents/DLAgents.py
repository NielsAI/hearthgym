import numpy as np
from numba import njit
from agents.HearthstoneAgent import HearthstoneAgent
from env.hearthstone.HearthGym import HearthstoneEnv

# Global constants and arrays
INT_MAX = np.iinfo(np.int32).max
INT_MIN = np.iinfo(np.int32).min
BASE_SCALING = np.array([21.5, 33.6, 41.1, 19.4, 54.0, 60.5, 88.5, 84.7])

@njit
def custom_score_num(hero_hp, op_hero_hp, hero_armor, op_armor, hero_atk,
                     own_count, opp_count, own_minion_health, own_minion_atk,
                     opp_minion_health, opp_minion_atk, base_scaling, runtime_scaling):
    # Lethal checks
    if op_hero_hp < 1:
        return INT_MAX
    if hero_hp < 1:
        return INT_MIN

    score = 0.0
    # Score hero's combined health and armor
    score += base_scaling[0] * runtime_scaling[0] * (hero_hp + hero_armor)
    score -= base_scaling[1] * runtime_scaling[1] * (op_hero_hp + op_armor)
    # Additional hero attack bonus
    score += 0.01 * hero_atk
    # Board count scoring
    score += base_scaling[2] * runtime_scaling[2] * own_count
    score -= base_scaling[3] * runtime_scaling[3] * opp_count
    # Minion health and attack scoring
    score += base_scaling[4] * runtime_scaling[4] * own_minion_health
    score += base_scaling[5] * runtime_scaling[5] * own_minion_atk
    score -= base_scaling[6] * runtime_scaling[6] * opp_minion_health
    score -= base_scaling[7] * runtime_scaling[7] * opp_minion_atk

    return int(round(score))

class CustomScore:
    """
    Custom scoring logic for the AI agent.
    """
    # Base scaling factors (provided as in C# code)
    BaseScaling = [21.5, 33.6, 41.1, 19.4, 54, 60.5, 88.5, 84.7]

    def __init__(self, controller):
        self.controller = controller

    def rate(self, runtime_scaling):
        # If no controller (e.g. during mulligan), return 0
        if self.controller is None:
            return 0

        # Extract numeric features from the controller
        hero_hp    = self.controller.hero.health
        op_hero_hp = self.controller.opponent.hero.health
        hero_armor = self.controller.hero.armor
        op_armor   = self.controller.opponent.hero.armor
        hero_atk   = self.controller.hero.atk

        board    = self.controller.field
        op_board = self.controller.opponent.field

        own_count = len(board)
        opp_count = len(op_board)

        own_minion_health = 0
        own_minion_atk    = 0
        for m in board:
            own_minion_health += m.health
            own_minion_atk    += m.atk

        opp_minion_health = 0
        opp_minion_atk    = 0
        for m in op_board:
            opp_minion_health += m.health
            opp_minion_atk    += m.atk

        # Convert runtime scaling to a numpy array (if not already)
        runtime_scaling_np = np.array(runtime_scaling)

        # Call the njitted function for a fast computation
        return custom_score_num(
            hero_hp, op_hero_hp,
            hero_armor, op_armor, hero_atk,
            own_count, opp_count,
            own_minion_health, own_minion_atk,
            opp_minion_health, opp_minion_atk,
            BASE_SCALING, runtime_scaling_np
            )

    def mulligan_rule(self, hand_cards):
        return [c.id for c in hand_cards if c.card.name != "Reno Jackson" and c.cost > 3]
    
    def mulligan_rule_mage(self, hand_cards):
        return [c.id for c in hand_cards if c.cost > 3]


class DynamicLookaheadAgent(HearthstoneAgent):
    DEFENSE_HEALTH_THRESHOLD = 1/3.0
    RuntimeScaling = [1.0]*8

    def __init__(self):
        super().__init__()
        self.sub_agent = None

    def act(self, observation, valid_actions=None, action_mask=None, env: HearthstoneEnv = None):
        # Extract game state from observation
        hero_class = int(observation[0])
                
        # Warrior = 10, Mage = 4      
        if hero_class == 10:
            self.sub_agent = self.WarriorSubAgent(self.RuntimeScaling)
        elif hero_class == 4:
            self.sub_agent = self.MageSubAgent(self.RuntimeScaling)

        if self.sub_agent is not None:
            return self.sub_agent.act(observation=observation, valid_actions=valid_actions, action_mask=action_mask, env=env)

        # Main agent logic (like the main block in C# code GetMove())
        player = env.game.current_player

        opponent = player.opponent
        options = player.hand

        # Example: If there's a "The Coin", play it
        coins = [(o, index) for index, o in enumerate(options) if o.name_enUS == "The Coin"]
        if coins:
            return {
                "action_type": 1,
                "card_index": coins[0][1],
                "attacker_index": 0,
                "target_index": 0,
                "discover_index": 0,
                "choose_index": 0
            }

        valid_opts = self.simulate(action_list=valid_actions, env=env)

        optcount = len(valid_opts)

        # Adjust runtime scaling based on health conditions
        if player.hero.health < self.DEFENSE_HEALTH_THRESHOLD * player.hero.max_health:
            self.RuntimeScaling[0] += 0.1
        if opponent.hero.health < self.DEFENSE_HEALTH_THRESHOLD * opponent.hero.health:
            self.RuntimeScaling[1] += 0.1
        
        if valid_opts:
            scored_actions = [self.score_action(opt, st, player.name, optcount) for opt, st in valid_opts]
            # Choose the action with the highest score
            
            # Remove the valid opts to free up memory
            valid_opts = None
            
            return max(scored_actions, key=lambda x: x[1])[0]
        else:
            # No valid options other than ending turn
            end_turn = [o for o in valid_actions if o[0] == 0]
            return end_turn[0] if end_turn else None

    def score_action(self, action, state, name, optcount, max_depth=3):
        def recursive_score(st, depth):
            max_score = -np.inf
            if depth > 0 and st.game.current_player.name == name:
                next_opts_list, _ = st.get_valid_actions()
                                        
                next_opts = self.simulate(action_list=next_opts_list, env=st)
                                        
                for sub_o, sub_s in next_opts:
                    candidate = recursive_score(sub_s, depth-1)
                    if candidate > max_score:
                        max_score = candidate
                
                # Remove the next opts to free up memory
                next_opts = None
                
            # Compare direct state score as well
            max_score = max(max_score, self.score_state(st, name))
            return max_score

        if optcount >= 25:
            depth = 1
        elif optcount >= 5:
            depth = 2
        else:
            depth = 3

        final_score = recursive_score(state, depth)
        return action, final_score

    def score_state(self, state, name):
        # Choose correct perspective
        controller = state.game.current_player if state.game.current_player.name == name else state.game.current_player.opponent
        return CustomScore(controller).rate(self.RuntimeScaling)

    def choose_task(self, task_type, player, choices):
        return {
            "type": task_type,
            "name": player.name,
            "choices": choices
        }

    class WarriorSubAgent(HearthstoneAgent):
        def __init__(self, runtime_scaling):
            super().__init__()
            self.RuntimeScaling = runtime_scaling

        def act(self, observation, valid_actions, action_mask, env=None):
            
            player = env.game.current_player

            options = player.hand
            if env.game.turn == 1:
                opt1 = [(o, index) for index, o in enumerate(options) if o.name_enUS == "N'Zoth's First Mate"]
                if opt1:
                    return {
                        "action_type": 1,
                        "card_index": opt1[0][1],
                        "attacker_index": 0,
                        "target_index": 0,
                        "discover_index": 0,
                        "choose_index": 0
                    }

            if env.game.turn == 3:
                opt2 = [(o, index) for index, o in enumerate(options) if o.name_enUS == "Friery War Axe"]
                if opt2:
                    return {
                        "action_type": 1,
                        "card_index": opt2[0][1],
                        "attacker_index": 0,
                        "target_index": 0,
                        "discover_index": 0,
                        "choose_index": 0
                    }

            if env.game.turn == 5:
                opt3 = [(o, index) for index, o in enumerate(options) if o.name_enUS == "Arcanite Reaper"]
                if opt3:
                    return {
                        "action_type": 1,
                        "card_index": opt3[0][1],
                        "attacker_index": 0,
                        "target_index": 0,
                        "discover_index": 0,
                        "choose_index": 0
                    }

            valid_opts = self.simulate(action_list=valid_actions, env=env)

            optcount = len(valid_opts)

            if valid_opts:
                scored_actions = [self.score_action(opt, st, player.name, optcount) for opt, st in valid_opts]
                
                # Remove the valid opts to free up memory
                valid_opts = None
                
                return max(scored_actions, key=lambda x: x[1])[0]
            else:
                # No valid options other than ending turn
                end_turn = [o for o in valid_actions if o[0] == 0]
                return end_turn[0] if end_turn else None

        def score_action(self, action, state, name, optcount):
            # Similar scoring logic as main
            def recursive_score(st, depth):
                max_score = -np.inf
                if depth > 0 and st.game.current_player.name == name:
                    next_opts_list, _ = st.get_valid_actions()
                
                    next_opts = self.simulate(action_list=next_opts_list, env=st)
                        
                    for sub_o, sub_s in next_opts:
                        candidate = recursive_score(sub_s, depth-1)
                        if candidate > max_score:
                            max_score = candidate
                            
                    # Remove the next opts to free up memory
                    next_opts = None
                            
                max_score = max(max_score, self.score_state(st, name))
                return max_score

            if optcount >= 25:
                depth = 1
            elif optcount >= 5:
                depth = 2
            else:
                depth = 3

            final_score = recursive_score(state, depth)
            return action, final_score

        def score_state(self, state, name):
            controller = state.game.current_player if state.game.current_player.name == name else state.game.current_player.opponent
            return CustomScore(controller).rate(self.RuntimeScaling)

        def _parent_choose_task(self, task_type, player, choices):
            # create a mulligan action or similar
            return {
                "type": task_type,
                "name": player.name,
                "choices": choices
            }

        class CustomScore(HearthstoneAgent.CustomScore):
            def __init__(self):
                super().__init__()
                pass


    class MageSubAgent(HearthstoneAgent):
        def __init__(self, runtime_scaling):
            super().__init__()
            self.RuntimeScaling = runtime_scaling

        def act(self, observation, valid_actions, action_mask, env=None):

            player = env.game.current_player

            opponent = player.opponent
            options = player.hand

            class1 = env.class1 
            class2 = env.class2

            valid_opts = self.simulate(action_list=valid_actions, env=env)

            optcount = len(valid_opts)

            # If Reno Jackson logic applies:
            reno_opts = [(o, index) for index, o in enumerate(options) if o.name_enUS == "Reno Jackson"]
            if reno_opts and (player.hero.health - opponent.hero.atk - sum(m.atk for m in opponent.field) <= 3 or player.hero.health < 10):
                # Try playing Reno and then consider subsequent options
                tmp_game = HearthstoneEnv(class1=class1, class2=class2, clone=env.game, card_data=env.card_data)
                
                mana_after_reno = max(0, tmp_game.current_player.remaining_mana - 6)
                # Filter subsequent moves after Reno
                valid_opts_loc, _ = tmp_game.get_valid_actions()
                
                valid_opts_filtered = self.simulate(action_list=valid_opts_loc, env=tmp_game)

                scored_loc = [self.score_action(o, s, player.name, len(valid_opts_filtered))
                              for o, s in valid_opts_filtered
                              if (o.source is None or o.source.cost <= mana_after_reno)]

                # Remove the valid opts to free up memory
                valid_opts_filtered = None

                # Filter out Fireblast if no mana
                scored_loc = [sc for sc in scored_loc if "Fireblast" not in str(sc[0]) or mana_after_reno >= 1]

                if scored_loc:
                    # Take best option after Reno
                    return max(scored_loc, key=lambda x: x[1])[0]

                # If no better option, just play Reno
                return reno_opts[0]

            if valid_opts:
                scored_actions = [self.score_action(opt, st, player.name, optcount) for opt, st in valid_opts]
                # Avoid playing Reno Jackson if possible
                scored_actions = [sa for sa in scored_actions if "Reno Jackson" not in str(sa[0])]
                
                # Remove the valid opts to free up memory
                valid_opts = None
                
                if not scored_actions:  # if all were Reno, fallback
                    scored_actions = [self.score_action(opt, st, player.name, optcount) for opt, st in valid_opts]
                return max(scored_actions, key=lambda x: x[1])[0]
            else:
                # No valid options other than ending turn
                end_turn = [o for o in valid_actions if o[0] == 0]
                return end_turn[0] if end_turn else None

        def score_action(self, action, state, name, optcount):
            def recursive_score(st, depth):
                max_score = -np.inf
                if depth > 0 and st.game.current_player.name == name:
                    next_opts_list, _ = st.get_valid_actions()
                
                    next_opts = self.simulate(action_list=next_opts_list, env=st)
                    
                    for sub_o, sub_s in next_opts:
                        candidate = recursive_score(sub_s, depth-1)
                        if candidate > max_score:
                            max_score = candidate
                            
                    # Remove the next opts to free up memory
                    next_opts = None
                            
                max_score = max(max_score, self.score_state(st, name))
                return max_score

            if optcount >= 25:
                depth = 1
            elif optcount >= 5:
                depth = 2
            else:
                depth = 3

            final_score = recursive_score(state, depth)
            return action, final_score

        def score_state(self, state, name):
            controller = state.game.current_player if state.game.current_player.name == name else state.game.current_player.opponent
            return CustomScore(controller).rate(self.RuntimeScaling)

        def _parent_choose_task(self, task_type, player, choices):
            return {
                "type": task_type,
                "name": player.name,
                "choices": choices
            }
            