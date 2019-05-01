# UCB MAB baseline
# UCB = empirical mean of specific arm + sqrt(2 log t/N_t), where N_t = number of trials of specific arm
import time, sys
from os.path import dirname, abspath
sys.path.append("/home/leduc/Deep-CFR/")
import numpy as np
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.game.AgentTournament import AgentTournament
from H2HEvaluator import H2HEval

class MAB:    
    
    def __init__(self, agent_list, test_agent, gamma = 1, n_hands = 2):
        self.agent_list = agent_list
        self.num_agents = len(self.agent_list)
        self.arm_times = [0 for _ in range(self.num_agents)]
        self.ucb = [0 for _ in range(self.num_agents)]
        self.rewards = [0 for _ in range(self.num_agents)]
        self.test_agent = test_agent
        self.gamma = gamma
        self.n_hands = n_hands 
        self.reward_list = [0]
        self.nash = []
        
    def run(self, n_episodes):
        nash_rewards = []
        variances = [0]
        nash_agent_path = "/home/leduc/poker_ai_data/eval_agent/SD-CFR_LEDUC_EXAMPLE_200/120/eval_agentAVRG_NET.pkl"
        nash_agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=nash_agent_path)
        time = np.sum(self.arm_times)
        
        def ucb_updater(time, arm_time):
            if (arm_time == 0):
                return np.infty
            else:
                return np.sqrt(2 * np.log(time) / np.float(arm_time))
            
        for _ in range(n_episodes):
            time += 1
            
            for i in range(self.num_agents):
                self.ucb[i] = self.rewards[i] + self.gamma * ucb_updater(time, self.arm_times[i])
            best_agent_idx = np.argmax(self.ucb)
            
            reward, variance = H2HEval(self.agent_list[best_agent_idx], self.test_agent).h2h_eval(self.n_hands)
            
            print("UCB List: ", self.ucb)
            self.arm_times[best_agent_idx] += 1
            self.rewards[best_agent_idx] = (self.rewards[best_agent_idx] * (self.arm_times[best_agent_idx] - 1) + reward) / self.arm_times[best_agent_idx]
            print("best agent, reward, arm_times, avg reward", best_agent_idx, reward,self.arm_times[best_agent_idx], self.rewards[best_agent_idx])
            self.reward_list.append((self.reward_list[-1] * (time - 1) + reward) / time)
            
            nash_rewards.append(H2HEval(nash_agent, self.test_agent).h2h_eval(self.n_hands)[0])
            self.nash.append(np.mean(nash_rewards))
            variances.append((variances[-1] * (time - 1) ** 2 + variance) / time ** 2)
            
        self.lower_bounds = [self.reward_list[i] - 1.96 * np.sqrt(variances[i]) for i in range(n_episodes)]
        self.upper_bounds = [self.reward_list[i] + 1.96 * np.sqrt(variances[i]) for i in range(n_episodes)]
        
        print("number of trials of each arm:", self.arm_times)