import copy
import numpy as np
import torch
import pickle

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

from RangeAgent import EvalAgentDeepRange
from PokerRL.game.games import StandardLeduc  # or any other game
from PokerRL.eval.rl_br.RLBRArgs import RLBRArgs

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

from PokerRL.game._.tree.PublicTree import PublicTree
from PokerRL.rl import rl_util

from RangeWrapper import RangeWrapper

NP_FLOAT_TYPE = np.float64  # Use 64 for extra stability in big games

class EvalAgentTree(_EvalAgentBase):
    
    def __init__(self, t_prof, br_agent, mode=None, device=None):
        super().__init__(t_prof=t_prof, mode=mode, device=device)
        
        self.tree = PublicTree(
            env_bldr=rl_util.get_env_builder(t_prof=t_prof),
            stack_size=t_prof.eval_stack_sizes[0],
            stop_at_street=None,
            put_out_new_round_after_limit=True,
            is_debugging=t_prof.DEBUGGING
        )
        self.tree.build_tree()
        self.br_agent = br_agent # agent to play best response against
        self.solve_br()
        
        self.modes = ["EVAL", "BR", "BAYESIAN"]
        if mode:
            self.mode = mode
        else:
            self.mode = "EVAL" # default is eval
            
        if self.mode == "BAYESIAN":
            self._fill_tree_w_prior()
                        
        
    def _fill_tree_w_prior(self, prior=1):
        def fill(node):
            node.data = np.ones((6,3)) #len(node.allowed_actions)
            for child in node.children:
                fill(child)

        node = self.tree.root
        fill(node)

        
    def can_compute_mode(self):
        """ All modes are always computable (i.e. not dependent on iteration etc.)"""
        return True
    
    def _find_node_by_env(self, action_history):
        node = self.tree.root
        
        """
        envw = self._internal_env_wrapper.env_bldr
        i = 0
        
        last_round_ = None
        round_ = node.env_state['current_round']
        p_id = node.p_id_acting_next
        
        if last_round_ == round_:
            nth_action_this_round += 1
        else:
            last_round_ = round_
            nth_action_this_round = 0
        
        def reverse_idx(idx, round_, p_id, nth_action_this_round):
            return i - nth_action_this_round - p_id * envw._VEC_HALF_ROUND_SIZE[round_] - envw._VEC_ROUND_OFFSETS[round_]
        
        while i < len(action_history):
            if action_history[i] == 1:
                action_idx = reverse_idx(i, round_, p_id, nth_action_this_round) + 1 # fold never accepted
                node = node.children[action_idx] #recurse through tree
        """
        
        i = 0
        
        while i < len(action_history):
            if isinstance(node.children[0], PlayerActionNode): #next node is playerAction
                action = action_history[i][0]
                assert(node.p_id_acting_next == action_history[i][2])
                node = node.children[node.allowed_actions.index(action)]
                i += 1
            else: #chance node, flop
                assert(node.children[0].action == "CHANCE")
                card = self._internal_env_wrapper.env.board
                node = node.children[self._card_to_idx(card)]
                assert(self._card_to_idx(node.env_state['board_2d']) == self._card_to_idx(card))
                
        if not isinstance(node.children[0], PlayerActionNode): # just need to do one more loop lol
            assert(node.children[0].action == "CHANCE")
            card = self._internal_env_wrapper.env.board
            node = node.children[self._card_to_idx(card)]
            assert(self._card_to_idx(node.env_state['board_2d']) == self._card_to_idx(card))
        
        return node
    
    def _card_to_idx(self, card):
        return card[0][0] * 2 + card[0][1]
    
    def solve_br(self):
        self.tree.fill_with_agent_policy(agent=self.br_agent)
        self.tree.compute_ev()
                    
    def get_action(self, step_env=True, need_probs=False):
        """ !! BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE !! """
        # print("action history", self._internal_env_wrapper._action_history_vector)
        # node = self._find_node_by_env(self._internal_env_wrapper._action_history_vector)
        
        # print("action history", self._internal_env_wrapper._action_history_list)
        node = self._find_node_by_env(self._internal_env_wrapper._action_history_list)
        
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        a_probs_all_hands = None
        
        if self.mode == "BR":
            action = None
            best_ev = -1e10 #really bad
            
            for idx, potential_action in enumerate(node.allowed_actions):
                if node.children[idx].ev[p_id_acting,range_idx] > best_ev:
                    action = potential_action # deterministic
                    best_ev = node.children[idx].ev[p_id_acting,range_idx]
            
        elif self.mode == "EVAL":
            a_probs = node.strategy[range_idx,:]
            # print(node.strategy, node.strategy.shape)
            # print(node.allowed_actions)
            # print("allowed:", legal_actions_list)
            # print(node.ev, node.ev_br)
            action = np.random.choice(node.allowed_actions, p=a_probs)
            
        elif self.mode == "BAYESIAN":
            
            psuedocounts = node.data[range_idx,node.allowed_actions]
            a_probs = psuedocounts/sum(psuedocounts)
            action = np.random.choice(node.allowed_actions, p=a_probs)
            
            """
            self.solve_br()
            action = None
            best_ev = -1e10 #really bad
            
            for idx, potential_action in enumerate(node.allowed_actions):
                if node.children[idx].ev[p_id_acting,range_idx] > best_ev:
                    action = potential_action # deterministic
                    best_ev = node.children[idx].ev[p_id_acting,range_idx]
            """

        if step_env:
            self._internal_env_wrapper.step(action=action)
        
        assert(a_probs_all_hands is None)
        
        return action, a_probs_all_hands
    
    def get_a_probs_for_each_hand(self):
        ## BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE!!!!!
        node = self._find_node_by_env(self._internal_env_wrapper._action_history_list)
        legal_actions_list = self._internal_env_wrapper.env.get_legal_actions()
        
        if self.mode == "BAYESIAN":
            x = node.data
            mask = np.ones(x.shape,dtype=bool) #np.ones_like(a,dtype=bool)
            mask[:,node.allowed_actions] = False
            x[mask] = 0
            return x / x.sum(axis=1)[:, np.newaxis]
        
    def get_mode(self):
        if self.mode == "BR":
            return "BESTRESPONSE"
        elif self.mode == "EVAL":
            return "COPYCAT"
        elif self.mode == "BAYESIAN":
            return "BAYESIAN"
            
