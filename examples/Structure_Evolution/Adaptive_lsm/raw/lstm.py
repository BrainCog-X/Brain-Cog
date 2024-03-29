import argparse, math, os, sys

from re import S
from aiohttp import ServerDisconnectedError
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt

from tools.ExperimentEnvGlobalNetworkSurvival import ExperimentEnvGlobalNetworkSurvival
from tools.MazeTurnEnvVec import MazeTurnEnvVec
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.utils as utils
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G')
parser.add_argument('--seed', type=int, default=598, metavar='N')
parser.add_argument('--num_steps', type=int, default=500, metavar='N')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N')
parser.add_argument('--render', action='store_true')
args = parser.parse_args()


n_agent = 1
steps = 500

env = MazeTurnEnvVec(n_agent, n_steps=steps)
data_env = ExperimentEnvGlobalNetworkSurvival(env)
s_dim = 4
a_dim = 3


class Policy(nn.Module):                                           
    def __init__(self, hidden_size, s_dim, a_dim):
        super(Policy, self).__init__()
        self.lstm = nn.LSTM(s_dim, hidden_size, batch_first = True)
        self.linear1 = nn.Linear(hidden_size, hidden_size)          
        self.linear2 = nn.Linear(hidden_size, a_dim)


    def forward(self, x,hidden):
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.linear1(x))
        p = F.softmax(self.linear2(x),-1)                            
        return p,hidden

class REINFORCE:
    def __init__(self, hidden_size, s_dim, a_dim):
        self.model = Policy(hidden_size, s_dim, a_dim)   
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2) # 
        self.model.train()
        self.pi = Variable(torch.FloatTensor([math.pi])) # 


    def select_action(self, state,hx,cx):
        # mu, sigma_sq = self.model(Variable(state).cuda())
        prob,(hx,cx) = self.model(Variable(state),(hx,cx))
        dist = Categorical(probs=prob)
        action = dist.sample()
        log_prob = prob[0][0,action.item()].log()
        # log_prob = prob.log()
        entropy = dist.entropy()
        
        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):# 更新参数
        R = torch.tensor(0)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]                                
            loss = loss - (log_probs[i]*Variable(R)) - 0.005*entropies[i][0]
            
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 2)             
        self.optimizer.step()

seeds=20
for seed in range(seeds):
    log_reward = []
    log_smooth = []
    gamma=np.linspace(0.9,1.0,100)
    for g in range(100):
        agent = REINFORCE(args.hidden_size,s_dim,a_dim)
        result=np.zeros([100,args.num_steps])
        for i_episode in range(args.num_episodes):
            state = torch.tensor(data_env.reset()).unsqueeze(0)
            entropies = []
            log_probs = []
            rewards = []
            old_dis = np.ones([1,])*13
            reawrd_perstep=[]
            allrewards=[]
            hx = torch.zeros(args.hidden_size).unsqueeze(0).unsqueeze(0)
            cx = torch.zeros(args.hidden_size).unsqueeze(0).unsqueeze(0)
            for t in range(args.num_steps): # 1个episode最长num_steps
                action, log_prob, entropy = agent.select_action(state.unsqueeze(0).float(),hx,cx)
                action = action.cpu().numpy()
                next_state, envreward, done, _ = data_env.step(action[0])
                entropies.append(entropy)
                log_probs.append(log_prob)
                state = torch.Tensor([next_state])

                rewards.append(envreward[0])
            agent.update_parameters(rewards, log_probs, entropies, gamma[g])

            print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

            log_reward.append(np.sum(rewards))
            if i_episode == 0:
                log_smooth.append(log_reward[-1])
            else:
                log_smooth.append(log_smooth[-1]*0.99+0.01*np.sum(rewards))

            plt.plot(log_smooth)
            plt.plot(log_reward)
            plt.pause(1e-5)
        result[g]=np.array(allrewards).squeeze(1)

    np.save('./lstm.npy',result)

