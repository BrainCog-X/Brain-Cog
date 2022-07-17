"""
Zoe Zhao 2022.5
ToM Demo
"""
import ast
import argparse
import time
import yaml
import copy
import abc
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

from tqdm import *
import matplotlib.pyplot as plt
import matplotlib
# import seaborn as sns
import pygame
pygame.init()
# sns.set(style='ticks', palette='Set2')
matplotlib.rcParams.update({'font.size': 12})
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from BrainArea.PFC_ToM import PFC_ToM
from BrainArea.TPJ import ToM
from BrainArea.dACC import *
from rulebasedpolicy.Find_a_way import *
from env.env import FalseBelief_env
import  sys

from BrainCog.base.connection import layer
from BrainCog.base.encoder.encoder import *
from BrainCog.base.node import node
from BrainCog.model_zoo.base_module import BaseLinearModule, BaseModule

#NPC2
#state
N_state = 6
cell_num = 6

# action
N_action = 5
NC=10 #50 cells represent one character

#synapstic
bfs = pow(cell_num, N_state) #before synapstic
afs = N_action * NC

#agent
C=10
A_state = 4
abfs = pow(cell_num, A_state) #agent before synapstic
aafs = N_action * C

parser = argparse.ArgumentParser(description='sequence character (policy inference)')
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--task', type=str, default='both')
parser.add_argument('--logdir', type=str, default='checkpoint')
parser.add_argument('--save_net_a', type=str, default='net_NPC_11.pth', help='save the parameters of net_agent')
parser.add_argument('--save_net_N', type=str, default='net_NPC_11.pth', help='save the parameters of net_NPC')
parser.add_argument('--device', default='cpu', help='device')  # cuda:0
parser.add_argument('--T', default=40, type=int, help='simulating time-steps')  # 模拟时长
parser.add_argument('--dt', default=1, type=int, help='simulating dt')  # 模拟dt
parser.add_argument('--episodes', default=25, type=int, help='episodes')
parser.add_argument('--trajectories', default=10, type=int, help='trajectories')
parser.add_argument('--greedy', default=0.8, type=int, help='exploration or exploitation')
parser.add_argument('--num_enpop', default=6, type=int, help='the number of one population in the encoding layer')  #
parser.add_argument('--num_depop', default=10, type=int, help='the number of one population in the decoding layer')  #
parser.add_argument('--num_stateA', default=2, type=int, help='the number of states')
parser.add_argument('--num_stateN', default=6, type=int, help='the number of states')
parser.add_argument('--num_action', default=5, type=int, help='the number of actions')
parser.add_argument('--reward', default=10, type=float, help='environment parameter reward')
args = parser.parse_args()


def update(env, net_agent_belief, net_NPC, episodes, trajectories):
    """
    agents learn to reach the goal without collision
    update agents' positions
    @param env:
    @param env1:
    @param net_agent_belief: the SNN network of agent
    @param net_NPC: the SNN network of NPC
    @param episodes: train times
    @return: None
    """
    for episode in tqdm(range(episodes)):
        timer = 0
        env.reset()
        env.actu_obs()
        scores = {
            'agent_0': 0,
            'NPC2_0' : 0,
            'agent_1': 0,
            'NPC2_1': 0,
        }
        Done_agent_0 = Done_agent_1 = False
        Done_NPC2_0  = Done_NPC2_1  = False
        action_agent  = 3
        action_NPC2   = 2
        action_NPC1   = 1
        action_agent1 = 4
        # the start position are the same in two envs
        # mapping_a = {'state': sum(env.agent['axis'], []),
        #              'action': action_agent}
        mapping_N = {'state': sum(env.NPC_2['axis'], []),
                     'action': action_NPC2}
        while True and timer < trajectories:
            timer = timer + 1
            NPC_1_state, NPC_2_state, Agent_state \
                = env.interact(action_NPC1, action_NPC2, action_agent)
            env.SHOW()
            # time.sleep(2)

            # NPC_1 selects action by pp
            if env.NPC_1['Done'] == False:
                action_seq1 = Find_a_way(size=5, board=NPC_1_state, \
                                         start_x=env.NPC_1['x'] - 1, \
                                         start_y=4 - (env.NPC_1['y'] - 1), \
                                         end_x=3, end_y=4 - 4)
                action_NPC1 = list(env.action_move.keys())[ \
                    list(env.action_move.values()).index(
                        (action_seq1[1][0] - (action_seq1[0][0]), -action_seq1[1][1] + (action_seq1[0][1])))]
            # agent selects action on purpose
            # Agent_obs = sum(env.agent['axis'], [])
            if env.agent['Done'] == False:
                axis_new, axis_switch, obs_switch = ToM.TPJ(NPC_num=2, axis=env.agent['axis'], obs=env.agent['obs'], )
                if axis_new == env.agent['axis']:
                    '''
                    没有遮挡关系 have teached
                    '''
                    action_agent = 3
                else:
                    '''
                    有遮挡关系
                    '''

                    Agent_obs_NPC2 = sum(env.NPC_2['axis'], [])
                    action_agent = net_agent_belief(inputs=Agent_obs_NPC2,
                                                    num_action=args.num_action,
                                                    episode=episode)
                    prediction_next_state = ToM.prediction_state(axis_new, env.agent['axis'], action_NPC1, net_NPC,
                                                                 num_action=args.num_action,
                                                                 episode = episode)
                    if ToM.state_evaluation(prediction_next_state=prediction_next_state) == False:
                        print(False)
                        action_agent = ToM.altruism(axis_switch=axis_switch , axis_NPC=env.NPC_2['axis'], n_actions = env.n_actions)
                        env.trigger = 1
                    else:
                        action_agent = 3
            # NPC_2 selects action by E-STDP
            NPC2_obs = sum(env.NPC_2['axis'], [])
            if Done_NPC2_0 == False:
                if action_agent == 4 and env.agent['Done'] == False:
                    action_NPC2 = 4
                else:
                    action_NPC2 = net_NPC(inputs=NPC2_obs, \
                                          num_action=args.num_action, \
                                          episode=episode)
                    state_NPC2 = copy.deepcopy(NPC2_obs)
                    Done_NPC2_0 = copy.deepcopy(env.NPC_2['Done'])
                    # mapping_N = {'state': state_NPC2,  # at time t
                    #              'action': action_NPC2}


def train():
    print('train mode loading ... ')

    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    bfs = pow(args.num_enpop, args.num_stateN)  # before synapstic
    afs = args.num_action * args.num_depop
    #agent
    # abfs = pow(args.num_enpop, args.num_stateA)  # agent before synapstic
    # aafs = args.num_action * args.num_depop
    net_agent_belief = PFC_ToM(step=args.T, encode_type='rate', bias=True,
                        in_features=bfs, out_features=afs,
                        node=node.LIFNode, num_state=args.num_stateN,
                        greedy=args.greedy)    #out_features the kinds of policies
    net_agent_belief.to(args.device)
    net_agent_belief.fc.weight.data = torch.rand((afs, bfs))
    # net_agent_belief.load_state_dict(torch.load(os.path.join(args.logdir, args.save_net_N))['model'])
    #NPC
    net_NPC = PFC_ToM(step=args.T, encode_type='rate', bias=True,
                     in_features=bfs, out_features=afs,
                     node=node.LIFNode, num_state=args.num_stateN,
                        greedy=args.greedy)    #out_features the kinds of policies
    net_NPC.to(args.device)

    net_NPC.load_state_dict(torch.load(os.path.join(args.logdir, args.save_net_N))['model'])
    total_scores = update(env, net_agent_belief, net_NPC, args.episodes,\
                          args.trajectories)

    # torch.save({'model': net_agent.state_dict()}, os.path.join(args.logdir, args.save_net_a))
    torch.save({'model': net_NPC.state_dict()}, os.path.join(args.logdir, args.save_net_N))

    time_end = time.time()
    print('totally cost',time_end-time_start)

if __name__ == "__main__":
    time_start = time.time()
    env = FalseBelief_env(args.reward)
    ToM = ToM(env=env)
    # args.task = 'both'#'zero'
    # args.mode = 'test'#'train'
    # args.save_net_N = 'net_NPC_3.pth'
    # args.save_net_a = 'net_agent_3.pth'
    # args.greedy = 111
    train()
