import copy
import numpy as np

import sys
sys.path.append("/home/leduc/Deep-CFR/")
sys.path.append("/home/leduc/PokerRL/")

from PokerRL.game import Poker
from PokerRL.game._.tree._.nodes import PlayerActionNode
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.EvalAgentBase import EvalAgentBase as _EvalAgentBase
from PokerRL.rl.errors import UnknownModeError

from DeepCFR.IterationStrategy import IterationStrategy
from DeepCFR.StrategyBuffer import StrategyBuffer
from DeepCFR.workers.la.AvrgWrapper import AvrgWrapper

from RangeWrapper import RangeWrapper

NP_FLOAT_TYPE = np.float64  # Use 64 for extra stability in big games

class EvalAgentDeepRange(_EvalAgentBase):
    
    def __init__(self, t_prof, mode=None, device=None):
        super().__init__(t_prof=t_prof, mode=mode, device=device)
        self.avrg_args = t_prof.module_args["avrg_training"]
        
        self.policy = RangeWrapper(avrg_training_args=self.avrg_args, owner=0, env_bldr=self.env_bldr, device=self.device)       
        self.policy._net.eval()

    def can_compute_mode(self):
        """ All modes are always computable (i.e. not dependent on iteration etc.)"""
        return True
    
    def _get_hand_info(self):
        norm_sum = float(sum([s.starting_stack_this_episode for s in self.seats])) / self.N_SEATS
        return np.array(
            self._get_table_state(normalization_sum=norm_sum) + self._get_board_state(), 
            dtype=np.float32
        )
    """
    def get_a_probs_for_each_hand(self):
        ## BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE!!!!!
        pub_obs = self._internal_env_wrapper.get_current_obs()
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id

        return self.policy.get_a_probs_for_each_hand(pub_obs=pub_obs, legal_actions_list=legal_actions_list)
    """
    def get_a_probs(self):
        pub_obs = self._internal_env_wrapper.get_current_obs()
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)

        return self.policy.get_a_probs(
            pub_obses=[pub_obs],
            range_idxs=np.array([range_idx], dtype=np.int32),
            legal_actions_lists=[legal_actions_list],
            hand_info=self._get_hand_info()
        )[0]

    def get_action(self, step_env=True, need_probs=False):
        """ !! BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE !! """

        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)

        if need_probs:  # only do if necessary
            a_probs_all_hands = self.get_a_probs_for_each_hand()
            a_probs = a_probs_all_hands[range_idx]
        else:
            a_probs_all_hands = None  # not needed

            a_probs = self.policy.get_a_probs(
                pub_obses=[self._internal_env_wrapper.get_current_obs()],
                range_idxs=np.array([range_idx], dtype=np.int32),
                legal_actions_lists=[self._internal_env_wrapper.env.get_legal_actions()],
                hand_info=self._get_hand_info()
            )[0]

        action = np.random.choice(np.arange(self.env_bldr.N_ACTIONS), p=a_probs)

        if step_env:
            self._internal_env_wrapper.step(action=action)
        
        assert(a_probs_all_hands is None)
        
        return action, a_probs_all_hands

    def get_action_frac_tuple(self, step_env):
        a_idx_raw = self.get_action(step_env=step_env, need_probs=False)[0]

        if self.env_bldr.env_cls.IS_FIXED_LIMIT_GAME:
            return a_idx_raw, -1
        else:
            if a_idx_raw >= 2:
                frac = self.env_bldr.env_args.bet_sizes_list_as_frac_of_pot[a_idx_raw - 2]
                return [Poker.BET_RAISE, frac]
            return [a_idx_raw, -1]

    def update_weights(self, weights_for_eval_agent):
        self.policy._net.load_net_state_dict(self.ray.state_dict_to_torch(weights_for_eval_agent,
                                                                                   device=self.device))
        self.policy._net.eval()

    def _state_dict(self):
        return self.policy._net.net_state_dict()

    def _load_state_dict(self, state):
        self.policy._net.load_net_state_dict(state)
        print("Net loaded")

