import numpy as np
from numba import njit
from agents.HearthstoneAgent import HearthstoneAgent
from env.hearthstone.HearthGym import HearthstoneEnv

# Global constants for lethal outcomes
INT_MAX = np.iinfo(np.int32).max
INT_MIN = np.iinfo(np.int32).min

@njit
def gretive_score_num(
    hero_hp, opp_hp, hero_atk, opp_atk, hero_armor, opp_hero_armor,
    hand_tot_cost, hand_count, opp_hand_count, deck_count, opp_deck_count,
    minion_total_atk, minion_total_health, opp_minion_total_atk, opp_minion_total_health,
    minion_taunt_health, opp_minion_taunt_health,
    W_HeroHp, W_OpHeroHp, W_HeroAtk, W_OpHeroAtk,
    W_HandCost, W_HandCount, W_OpHandCount, W_DeckCount, W_OpDeckCount,
    W_MinionTotalAtk, W_MinionTotalHth, W_OpMinionTotalAtk, W_OpMinionTotalHth,
    W_MinionTauntHealth, W_OpMinionTauntHealth, W_HeroArmor, W_OpHeroArmor
    ):
    # Lethal checks:
    if hero_hp < 1:
        return INT_MIN
    if opp_hp < 1:
        return INT_MAX

    # Compute weighted sum of all features.
    value = (W_HeroHp * hero_hp + W_OpHeroHp * opp_hp +
             W_HeroAtk * hero_atk + W_OpHeroAtk * opp_atk +
             W_HandCost * hand_tot_cost + W_HandCount * hand_count +
             W_OpHandCount * opp_hand_count + W_DeckCount * deck_count +
             W_OpDeckCount * opp_deck_count + W_MinionTotalAtk * minion_total_atk +
             W_MinionTotalHth * minion_total_health + W_OpMinionTotalAtk * opp_minion_total_atk +
             W_OpMinionTotalHth * opp_minion_total_health + W_MinionTauntHealth * minion_taunt_health +
             W_OpMinionTauntHealth * opp_minion_taunt_health + W_HeroArmor * hero_armor +
             W_OpHeroArmor * opp_hero_armor)
    return int(round(value))

class Profile:
    AGGRO = 1
    MIDRANGE = 2
    CONTROL = 3
    DEFAULT_BY_HERO = 4

class MidRangeScore:
    def mulligan_rule(self, hand_cards):
        return [c.id for c in hand_cards if c.cost > 3]

class GretiveScore:
    def __init__(self):
        # Weights – these will be set by the dispatcher.
        self.W_HeroHp = 0
        self.W_OpHeroHp = 0
        self.W_HeroAtk = 0
        self.W_OpHeroAtk = 0
        self.W_HandCost = 0
        self.W_HandCount = 0
        self.W_OpHandCount = 0
        self.W_DeckCount = 0
        self.W_OpDeckCount = 0
        self.W_MinionTotalAtk = 0
        self.W_MinionTotalHth = 0
        self.W_OpMinionTotalAtk = 0
        self.W_OpMinionTotalHth = 0
        self.W_MinionTauntHealth = 0
        self.W_OpMinionTauntHealth = 0
        self.W_HeroArmor = 0
        self.W_OpHeroArmor = 0

        # This will be set to the current player's controller before scoring.
        self.Controller = None

    def check_victory(self):
        return self.Controller.opponent.hero.health < 1

    def check_defeat(self):
        return self.Controller.hero.health < 1

    def rate(self):
        if self.Controller is None:
            return 0

        # Extract hero metrics
        hero_hp    = self.Controller.hero.health
        opp_hp     = self.Controller.opponent.hero.health
        hero_atk   = self.Controller.hero.atk
        opp_atk    = self.Controller.opponent.hero.atk
        hero_armor = self.Controller.hero.armor
        opp_armor  = self.Controller.opponent.hero.armor

        # Extract hand and deck data
        hand_cards      = self.Controller.hand
        opp_hand_cards  = self.Controller.opponent.hand
        deck_count      = len(self.Controller.deck)
        opp_deck_count  = len(self.Controller.opponent.deck)
        hand_tot_cost   = 0
        for card in hand_cards:
            hand_tot_cost += card.cost
        hand_count     = len(hand_cards)
        opp_hand_count = len(opp_hand_cards)

        # Extract board data (minion stats)
        board      = self.Controller.field
        opp_board  = self.Controller.opponent.field

        minion_total_atk    = 0
        minion_total_health = 0
        for m in board:
            minion_total_atk    += m.atk
            minion_total_health += m.health

        opp_minion_total_atk    = 0
        opp_minion_total_health = 0
        for m in opp_board:
            opp_minion_total_atk    += m.atk
            opp_minion_total_health += m.health

        # Extract taunt-specific totals
        minion_taunt_health    = 0
        for m in board:
            if m.taunt:
                minion_taunt_health += m.health
        opp_minion_taunt_health = 0
        for m in opp_board:
            if m.taunt:
                opp_minion_taunt_health += m.health

        # Delegate to the jitted scoring function.
        return gretive_score_num(
            hero_hp, opp_hp, hero_atk, opp_atk, hero_armor, opp_armor,
            hand_tot_cost, hand_count, opp_hand_count, deck_count, opp_deck_count,
            minion_total_atk, minion_total_health, opp_minion_total_atk, opp_minion_total_health,
            minion_taunt_health, opp_minion_taunt_health,
            self.W_HeroHp, self.W_OpHeroHp, self.W_HeroAtk, self.W_OpHeroAtk,
            self.W_HandCost, self.W_HandCount, self.W_OpHandCount,
            self.W_DeckCount, self.W_OpDeckCount,
            self.W_MinionTotalAtk, self.W_MinionTotalHth, self.W_OpMinionTotalAtk, self.W_OpMinionTotalHth,
            self.W_MinionTauntHealth, self.W_OpMinionTauntHealth,
            self.W_HeroArmor, self.W_OpHeroArmor
        )

class GretiveDispatcher:
    @staticmethod
    def score(hero_class, hero_profile):
        if hero_class == 2:  # Druid
            return GretiveDispatcher.gretive_score_gen(32.65355, -47.92743, 35.84834, -48.95403,
                                                       21.174, -29.30753, 52.53054, -18.79501,
                                                       27.23606, -39.4745)
        elif hero_class == 3:  # Hunter
            return GretiveDispatcher.gretive_score_gen(34.0909, -48.18912, 26.89414, -33.90081,
                                                       11.47699, -44.57977, 30.48743, -46.80523,
                                                       19.92627, -42.86041)
        elif hero_class == 4:  # Mage
            return GretiveDispatcher.gretive_score_gen(38.80242, -92.0281, 23.84993, -22.77378,
                                                       13.82049, -52.32899, 43.57949, -45.75875,
                                                       14.2201, -27.63297)
        elif hero_class == 5:  # Paladin
            return GretiveDispatcher.gretive_score_gen(48.19408, -90.89896, 22.04053, -40.43081,
                                                       15.6046, -35.75431, 48.71384, -14.92025,
                                                       44.28719, -47.92072)
        elif hero_class == 6:  # Priest
            return GretiveDispatcher.gretive_score_gen(44.85069, -48.65612, 35.61387, -42.27365,
                                                       18.88283, -94.16244, 20.50913, -36.56719,
                                                       6.203781, -20.33816)
        elif hero_class == 7:  # Rogue
            return GretiveDispatcher.gretive_score_gen(31.03984, -41.27439, 36.42944, -13.64148,
                                                       53.96317, -32.19125, 8.117135, -48.25733,
                                                       26.03877, -36.07202)
        elif hero_class == 8:  # Shaman
            return GretiveDispatcher.gretive_score_gen(38.80242, -92.0281, 23.84993, -22.77378,
                                                       13.82049, -52.32899, 43.57949, -45.75875,
                                                       14.2201, -27.63297)
        elif hero_class == 9:  # Warlock
            return GretiveDispatcher.gretive_score_gen(38.80242, -92.0281, 23.84993, -22.77378,
                                                       13.82049, -52.32899, 43.57949, -45.75875,
                                                       14.2201, -27.63297)
        elif hero_class == 10:  # Warrior
            return GretiveDispatcher.gretive_score_gen(42.93896, -42.83433, 12.74382, -26.04122,
                                                       31.2325, -31.81688, 50.74067, -38.31769,
                                                       29.32415, -32.67503)
        else:
            # Default
            return GretiveDispatcher.gretive_score_gen(32.65355, -47.92743, 35.84834, -48.95403,
                                                       21.174, -29.30753, 52.53054, -18.79501,
                                                       27.23606, -39.4745)

    @staticmethod
    def gretive_score_gen(minTotalAtk, opMinTotalAtk, minTotalHth, opMinTotalHth,
                           minTauntHth, opMinTauntHth, heroArmor, opHeroArmor, heroHp, opHeroHp):
        score = GretiveScore()
        score.W_MinionTotalAtk = minTotalAtk
        score.W_OpMinionTotalAtk = opMinTotalAtk
        score.W_MinionTotalHth = minTotalHth
        score.W_OpMinionTotalHth = opMinTotalHth
        score.W_MinionTauntHealth = minTauntHth
        score.W_OpMinionTauntHealth = opMinTauntHth
        score.W_HeroArmor = heroArmor
        score.W_OpHeroArmor = opHeroArmor
        score.W_HeroHp = heroHp
        score.W_OpHeroHp = opHeroHp
        return score

class GretiveCompAgent(HearthstoneAgent):
    def __init__(self):
        super().__init__()
        self._score = None
        self._initialized = False
        self._player_name = None  # This will store the perspective player's name

    def act(self, observation, valid_actions=None, action_mask=None, env: HearthstoneEnv = None):
        # Extract hero_class from observation (adjust indexing as needed)
        hero_class = int(observation[0])
        # Use a default profile for this example
        hero_profile = Profile.DEFAULT_BY_HERO

        if not self._initialized:
            self.init_by_hero(hero_class, hero_profile)
            self._player_name = env.game.current_player.name

        player = env.game.current_player

        valid_opts = self.simulate(action_list=valid_actions, env=env)
        if not valid_opts:
            end_turn = [o for o in valid_actions if o[0] == 0]
            return end_turn[0] if end_turn else None

        count = len(valid_opts)
        depth = 1 if count > 32 else (3 if (count < 12 or (env.game.turn > 12 and count < 22)) else 2)

        scored = [self.tree_search((opt, st), depth) for opt, st in valid_opts]
        valid_opts = None  # Free memory
        best = max(scored, key=lambda x: x[1])[0]
        return best

    def init_by_hero(self, hero_class, profile=Profile.DEFAULT_BY_HERO):
        self._score = GretiveDispatcher.score(hero_class, profile)
        self._initialized = True

    def rate(self, env_state):
        game = env_state.game
        controller = game.current_player if game.current_player.name == self._player_name else game.current_player.opponent
        self._score.Controller = controller
        return self._score.rate()

    def tree_search(self, game_state_pair, depth=2):
        action, env_state = game_state_pair
        if depth == 0 or env_state.game.current_player.name != self._player_name:
            return (action, self.rate(env_state))

        next_opts_list, _ = env_state.get_valid_actions()
        if not next_opts_list:
            return (action, self.rate(env_state))

        class1 = env_state.class1
        class2 = env_state.class2

        best_value = INT_MIN
        for sub_action in next_opts_list:
            env_copy = HearthstoneEnv(class1=class1, class2=class2, clone=env_state.game, card_data=env_state.card_data)
            # Assume env_copy.step() advances the state
            _, _, _, _, _ = env_copy.step(sub_action)
            val = self.tree_search((sub_action, env_copy), depth - 1)[1]
            if val > best_value:
                best_value = val

        return (action, best_value)

    def choose_task(self, task_type, player, choices):
        return {
            "type": task_type,
            "name": player.name,
            "choices": choices
        }
