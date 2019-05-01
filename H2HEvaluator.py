import time, sys
from os.path import dirname, abspath

sys.path.append("/home/leduc/Deep-CFR/")

import numpy as np

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.AgentTournament import AgentTournament
from DebugAgentTournament import DebugAgentTournament

# Copyright (c) 2019 Eric Steinberger


import numpy as np

class H2HEval:
    def __init__(self, agent1, agent2):
        self.eval_agent_1 = agent1
        self.eval_agent_2 = agent2
    
    def h2h_eval(self, n_games = 100):
        eval_agent_1 = self.eval_agent_1
        eval_agent_2 = self.eval_agent_2
        # assert eval_agent_dcfr.t_prof.name == eval_agent_sdcfr.t_prof.name

        start_time = time.time()

        env_bldr = eval_agent_1.env_bldr
        env = env_bldr.get_new_env(is_evaluating=False)
        env_cls = env_bldr.env_cls
        env_args = env_bldr.env_args

        # print("Agent 1:", eval_agent_1.t_prof.name)
        # print("Agent 2:", eval_agent_2.t_prof.name)
        matchup = AgentTournament(env_cls, env_args, eval_agent_1, eval_agent_2)
        # matchup = DebugAgentTournament(env_cls, env_args, eval_agent_1, eval_agent_2)
        mean, upper_conf95, lower_conf95 = matchup.run(n_games_per_seat=n_games)

        end_time = time.time()

        # print("Time taken", end_time - start_time)
        # print("mean", mean)
              
        return mean, ((upper_conf95 - mean) / 1.96) ** 2