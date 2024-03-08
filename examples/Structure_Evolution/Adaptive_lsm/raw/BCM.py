import argparse, math, os, sys
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import nsganet as engine
from pymop.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
from tools.ExperimentEnvGlobalNetworkSurvival import ExperimentEnvGlobalNetworkSurvival
from tools.MazeTurnEnvVec import MazeTurnEnvVec
import torch
import torch.nn.utils as utils
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from itertools import product

from functools import partial
import torchvision, pprint
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.model_zoo.base_module import BaseModule
from braincog.base.learningrule.BCM import *
from braincog.base.learningrule.STDP import *

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G')
parser.add_argument('--seed', type=int, default=1, metavar='N')
parser.add_argument('--num_steps', type=int, default=500, metavar='N')
parser.add_argument('--num_episodes', type=int, default=100, metavar='N')
parser.add_argument('--render', action='store_true')
args = parser.parse_args()


n_agent = 1
steps = 500
hidden_size=64
env = MazeTurnEnvVec(n_agent, n_steps=steps)
data_env = ExperimentEnvGlobalNetworkSurvival(env)
s_dim = 4
a_dim = 3
def randbool(size, p):
    return torch.rand(*size) < p


def fit(agent):
    states = list(product([0, 1], repeat=4))
    ls_list=[]
    for state in states:
        agent.model.reset()
        state_tensor = torch.tensor(state).float().reshape(1, -1)
        la, ls = agent.model(Variable(state_tensor.float()).reshape(-1,)) 
        ls_np = ls.detach().numpy() 
        ls_list.append(ls_np)

    ls_matrix = np.vstack(ls_list)

    rank = np.linalg.matrix_rank(ls_matrix)

    return rank

@register_model
class SNN(BaseModule):                                        
    def __init__(self,
                 hidden_size,
                 n_agent,
                 connectivity_matrix,
                 num_classes=3,
                 step=1,
                 node_type=LIFNode,
                 encode_type='direct',
                 ins=4,
                 lsm_th=0.3,
                 fc_th=0.3,
                 lsm_tau=3,
                 fc_tau=3,
                 tw=100,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.linear1 = nn.Linear(s_dim, hidden_size)      
        self.node=partial(node_type, **kwargs, step=step,tau=lsm_tau,threshold=lsm_th)    
        self.linear2 = nn.Linear(hidden_size, a_dim)

        self.node_lsm=partial(node_type, **kwargs, step=step,tau=lsm_tau,threshold=lsm_th)
        self.node_fc = partial(node_type, **kwargs, step=step,tau=fc_tau,threshold=fc_th)
        self.hidden_size=hidden_size
        self.out = torch.zeros(hidden_size)
        self.con=[]
        self.learning_rule=[]
        self.connectivity_matrix=connectivity_matrix
        w1tmp=nn.Linear(ins,hidden_size,bias=False)
        self.con.append(w1tmp)
        w2tmp=nn.Linear(hidden_size,hidden_size,bias=False)
        self.liquid_weight=w2tmp.weight.data
        w2tmp.weight.data=w2tmp.weight.data*self.connectivity_matrix
        self.con.append(w2tmp)
        self.learning_rule.append(BCM(self.node_lsm(), [self.con[0], self.con[1]])) 
        self.fc = nn.Linear(hidden_size,num_classes)
        self.learning_rule.append(BCM(self.node_fc(), [self.fc])) 


    def forward(self, x):
        sum_spike=0
        time_window=20
        self.tw=time_window
        self.firing_tw=torch.zeros(time_window, self.hidden_size)
        self.out = torch.zeros(self.hidden_size)
        for t in range(time_window):
            self.out, self.dw = self.learning_rule[0](x, self.out)
            self.con[1].weight.data+=self.dw[1]
            out_liquid=self.out[0:self.hidden_size]
            xout,dw = self.learning_rule[1](out_liquid)
            self.fc.weight.data+=dw[0]
            sum_spike=sum_spike+xout
            self.firing_tw[t]=out_liquid
        outputs = sum_spike+0.0001 / time_window
        return outputs,out_liquid

class REINFORCE:
    def __init__(self, lm):
        self.model = SNN(ins=4,n_agent=n_agent,hidden_size=hidden_size,lsm_tau=2,lsm_th=0.2,connectivity_matrix=lm)
        self.model.train()


    def select_action(self, state):
        # mu, sigma_sq = self.model(Variable(state).cuda())
        prob,_= self.model(Variable(state).reshape(-1,))
        dist = Categorical(probs=prob)
        action = dist.sample()
        log_prob = prob[action.item()].log()
        entropy = dist.entropy()
        return action, log_prob, entropy




class Evolve(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int64)
        self.xl = lb
        self.xu = ub
        self._n_evaluated = 0  # keep track of how many architectures are sampled


    def _evaluate(self, x, out, *args, **kwargs):
        
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('Network= {}'.format(arch_id))

            agent = REINFORCE(torch.from_numpy(x[i].reshape(hidden_size,hidden_size)).float())
            log_reward = []
            log_smooth = []
            # gamma=np.linspace(0.9,1.0,100)
            gam=0.9
            # for gam in gamma:
            for i_episode in range(100):
                state = torch.tensor(data_env.reset()).unsqueeze(0)
                entropies = []
                log_probs = []
                rewards = []
                old_dis = np.ones([1,])*13
                reawrd_perstep=[]
                ss=0
                allrewards=[]
                for t in range(500): 
                    action, log_prob, entropy = agent.select_action(state.float())
                    action=action.unsqueeze(0).numpy()
                    next_state, envreward, done, _ = data_env.step(action)
                    entropies.append(entropy)
                    log_probs.append(log_prob)
                    state = torch.Tensor([next_state])
                    rewards.append(envreward[0])
                print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
                log_reward.append(np.sum(rewards))
                if i_episode == 0:
                    log_smooth.append(log_reward[-1])
                else:
                    log_smooth.append(log_smooth[-1]*0.99+0.01*np.sum(rewards))
                plt.plot(log_smooth)
                plt.plot(log_reward)
                plt.pause(1e-5)

            objs[i, 0] = fit(agent)
            self._n_evaluated += 1
        out["F"] = objs

def do_every_generations(algorithm):
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

if __name__ == "__main__":
    n_agent=1
    kkk = Evolve(n_var=hidden_size*hidden_size, 
                    n_obj=1, n_constr=0)
    method = engine.nsganet(pop_size=n_agent,
                            sampling=RandomSampling(var_type='custom'),
                            mutation=BinaryBitflipMutation(),
                            n_offsprings=10,
                            eliminate_duplicates=True)
    kres=minimize(kkk,
                    method,
                    callback=do_every_generations,
                    termination=('n_gen', 1000))
    
