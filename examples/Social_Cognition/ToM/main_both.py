import argparse
import time
import copy
import numpy as np
import torch
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)
from tqdm import *
import matplotlib
import seaborn as sns
import pygame
pygame.init()
sns.set(style='ticks', palette='Set2')
matplotlib.rcParams.update({'font.size': 12})
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from BrainArea.PFC_ToM import PFC_ToM
from rulebasedpolicy.Find_a_way import *
from env.env3_train_env00 import FalseBelief_env0   #3
from env.env3_train_env01 import FalseBelief_env1   #2
from braincog.base.encoder.encoder import *
from braincog.base.node import node
torch.manual_seed(1)
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
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--task', type=str, default='both')
parser.add_argument('--logdir', type=str, default='checkpoint')
parser.add_argument('--save_net_a', type=str, default='net_agent_4.pth', help='save the parameters of net_agent')
parser.add_argument('--save_net_N', type=str, default='net_NPC_4.pth', help='save the parameters of net_NPC')
parser.add_argument('--device', default='cpu', help='device')  # cuda:0
parser.add_argument('--T', default=40, type=int, help='simulating time-steps')  # 模拟时长
parser.add_argument('--dt', default=1, type=int, help='simulating dt')  # 模拟dt
parser.add_argument('--episodes', default=25, type=int, help='episodes')
parser.add_argument('--trajectories', default=10, type=int, help='trajectories')
parser.add_argument('--greedy', default=0.8, type=float, help='exploration or exploitation')
parser.add_argument('--num_enpop', default=6, type=int, help='the number of one population in the encoding layer')  #
parser.add_argument('--num_depop', default=10, type=int, help='the number of one population in the decoding layer')  #
parser.add_argument('--num_stateA', default=2, type=int, help='the number of states, (X, Y)')
parser.add_argument('--num_stateN', default=6, type=int, help='the number of states, [(X, Y), (X, Y), (X, Y)]')
parser.add_argument('--num_action', default=5, type=int, help='the number of actions')
parser.add_argument('--reward', default=10, type=float, help='environment parameter reward')
args = parser.parse_args()

def reward_plot(episodes, scores, Note):
    fig = plt.figure(figsize=(7.5, 4.5))
    ax1 = fig.add_subplot(111)
    ax1.set_title('Reward Plot')
    plt.xlim(1, episodes)
    plt.grid(ls='--', c='gray')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    episodes_list = list(range(1,episodes+1))
    plt.plot(episodes_list, scores['be observed agent without the ToM'], label='be observed agent without the ToM')
    plt.legend()
    plt.savefig('reward_plot_' + str(episodes) + '.png')

def update(env0, env1, net_agent, net_NPC, episodes, trajectories, task):
    """
    agents learn to reach the goal without collision
    update agents' positions
    @param env0:
    @param env1:
    @param net_agent: the SNN network of agent
    @param net_NPC: the SNN network of NPC
    @param episodes: train times
    @return: None
    """
    scores_agent = []
    scores_NPC2  = []
    for episode in tqdm(range(episodes)):
        timer0 = 0
        timer1 = 0
        env0.reset()
        env1.reset()
        env0.actu_obs()
        env1.actu_obs()
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
        action_NPC1   = 4
        action_agent1 = 4
        # the start position are the same in two envs
        mapping_a = {'state': sum(env0.agent['axis'], []),
                     'action': action_agent}
        mapping_N = {'state': sum(env0.NPC_2['axis'], []),
                     'action': action_NPC2}
        if task == 'both' or task == 'zero':
            while True and timer0 < trajectories:
                timer0 = timer0 + 1
                NPC_1_state, NPC_2_state, Agent_state \
                    = env0.interact(action_NPC1, action_NPC2, action_agent)
                env0.SHOW()
                # time.sleep(2)
                #NPC_1 selects action by pp
                if env0.NPC_1['Done'] == False:
                    action_seq1 = Find_a_way(size=5, board=NPC_1_state,\
                                             start_x=env0.NPC_1['x']-1,\
                                             start_y=4-(env0.NPC_1['y']-1),\
                                             end_x=3, end_y=4-4)
                    action_NPC1 = list(env0.action_move.keys())[\
                        list(env0.action_move.values()).index((action_seq1[1][0]-(action_seq1[0][0]), -action_seq1[1][1]+(action_seq1[0][1])))]
                #agent selects action by E-STDP
                Agent_obs = sum(env0.agent['axis'], [])
                if Done_agent_0 == False:
                    action_agent = 3
                    # net_agent.update_s(R = env0.agent['reward'],\
                    #                    mapping=mapping_a)
                    # action_agent = net_agent(inputs = Agent_obs,\
                    #                          num_action = args.num_action,\
                    #                          episode = episode)
                    # state_agent = copy.deepcopy(Agent_obs)
                    # Done_agent_0 = copy.deepcopy(env0.agent['Done'])
                    # mapping_a = {'state': state_agent, # at time t
                    #            'action': action_agent}
                #NPC_2 selects action by E-STDP
                NPC2_obs = sum(env0.NPC_2['axis'], [])
                if Done_NPC2_0 == False:
                    net_NPC.update_s(R = env0.NPC_2['reward'], \
                                       mapping=mapping_N)
                    action_NPC2 = net_NPC(inputs = NPC2_obs,\
                                          num_action = args.num_action,\
                                             episode = episode)
                    state_NPC2 = copy.deepcopy(NPC2_obs)
                    Done_NPC2_0 = copy.deepcopy(env0.NPC_2['Done'])
                    mapping_N = {'state': state_NPC2, # at time t
                               'action': action_NPC2}
                    # continue
                scores['agent_0'] += env0.agent['reward']
                scores['NPC2_0'] += env0.NPC_2['reward']
                if env0.NPC_1['Done'] == env0.NPC_2['Done'] == env0.agent['Done'] == True:
                    break
            scores_agent.append(scores['agent_0'])
            scores_NPC2.append(scores['NPC2_0'])
######################
        if task == 'both' or task == 'one':
            while True and timer1 < trajectories:
                timer1 = timer1 + 1
                NPC_2_state, Agent_state \
                    = env1.interact(action_NPC2, action_agent)
                env1.SHOW()
                # time.sleep(2)
                # agent selects action by E-STDP
                Agent_obs = sum(env1.agent['axis'], [])
                if Done_agent_1 == False:
                    action_agent = 3
                    # net_agent.update_s(R=env1.agent['reward'], \
                    #                    mapping=mapping_a)
                    # scores['agent_1'] += env1.agent['reward']
                    # action_agent = net_agent(inputs=Agent_obs, \
                    #                          num_action=args.num_action,\
                    #                          episode = episode)
                    # state_agent = copy.deepcopy(Agent_obs)
                    # Done_agent_1 = copy.deepcopy(env1.agent['Done'])
                    # mapping_a = {'state': state_agent,  # at time t
                    #              'action': action_agent}
                # NPC_2 selects action by E-STDP
                NPC2_obs = sum(env1.NPC_2['axis'], [])
                if Done_NPC2_1 == False:
                    net_NPC.update_s(R=env1.NPC_2['reward'], \
                                     mapping=mapping_N)
                    scores['NPC2_1'] += env1.NPC_2['reward']
                    action_NPC2 = net_NPC(inputs=NPC2_obs, \
                                          num_action=args.num_action,\
                                             episode = episode)
                    state_NPC2 = copy.deepcopy(NPC2_obs)
                    Done_NPC2_1 = copy.deepcopy(env1.NPC_2['Done'])
                    mapping_N = {'state': state_NPC2,  # at time t
                                 'action': action_NPC2}
                scores['agent_1'] += env1.agent['reward']
                scores['NPC2_1'] += env1.NPC_2['reward']
                if env1.NPC_2['Done'] == env1.agent['Done'] == True:
                    break
            scores_agent.append(scores['agent_1'])
            scores_NPC2.append(scores['NPC2_1'])
    total_scores = {
        'the agent with the ToM': scores_agent,
        'be observed agent without the ToM' : scores_NPC2
    }
    return total_scores

def train():
    print('train mode loading ... ')
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)
    #agent
    abfs = pow(args.num_enpop, args.num_stateA)  # agent before synapstic
    aafs = args.num_action * args.num_depop
    net_agent = PFC_ToM(step=args.T, encode_type='rate', bias=True,
                        in_features=abfs, out_features=aafs,
                        node=node.LIFNode, num_state=args.num_stateA,
                        greedy=args.greedy)    #out_features the kinds of policies
    net_agent.to(args.device)
    net_agent.fc.weight.data = torch.rand((aafs, abfs))
    # net_agent.load_state_dict(torch.load('./checkpoint/net_agent_12.pth')['model'])
    #NPC
    bfs = pow(args.num_enpop, args.num_stateN)  # before synapstic
    afs = args.num_action * args.num_depop
    net_NPC = PFC_ToM(step=args.T, encode_type='rate', bias=True,
                     in_features=bfs, out_features=afs,
                     node=node.LIFNode, num_state=args.num_stateN,
                        greedy=args.greedy)    #out_features the kinds of policies
    net_NPC.to(args.device)
    net_NPC.fc.weight.data = torch.rand((afs, bfs))
    # net_NPC.load_state_dict(torch.load('./checkpoint/net_NPC_12.pth')['model'])
    total_scores = update(env0, env1, net_agent, net_NPC, args.episodes,\
                          args.trajectories, args.task)
    torch.save({'model': net_agent.state_dict()}, os.path.join(args.logdir, args.save_net_a))
    torch.save({'model': net_NPC.state_dict()}, os.path.join(args.logdir, args.save_net_N))
    time_end = time.time()
    print('totally cost',time_end-time_start)
    if args.task == 'zero' or args.task == 'one':
        reward_plot(args.episodes, total_scores, 'Scores')
    elif args.task == 'both':
        reward_plot(args.episodes * 2, total_scores, 'Scores')
    plt.show()

def test():
    args.greedy = 1
    print('test mode loading ... ')
    print('greedy :', args.greedy)
    #agent
    abfs = pow(args.num_enpop, args.num_stateA)  # agent before synapstic
    aafs = args.num_action * args.num_depop
    net_agent = PFC_ToM(step=args.T, encode_type='rate', bias=True,
                     in_features=abfs, out_features=aafs,
                     node=node.LIFNode, num_state=args.num_stateA,
                          greedy=args.greedy)
    net_agent.to(args.device)
    # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    net_agent.load_state_dict(torch.load(os.path.join(args.logdir, args.save_net_a))['model'])      #out_features the kinds of policies
    #NPC
    bfs = pow(args.num_enpop, args.num_stateN)  # before synapstic
    afs = args.num_action * args.num_depop
    net_NPC = PFC_ToM(step=args.T, encode_type='rate', bias=True,
                     in_features=bfs, out_features=afs,
                     node=node.LIFNode, num_state=args.num_stateN,
                        greedy=args.greedy)
    net_NPC.to(args.device)
    net_NPC.load_state_dict(torch.load(os.path.join(args.logdir, args.save_net_N))['model'])   #out_features the kinds of policies
    total_scores = update(env0, env1, net_agent, net_NPC, args.episodes,
                          args.trajectories, args.task)
    time_end = time.time()
    print('totally cost',time_end-time_start)
    if args.task == 'zero' or args.task == 'one':
        reward_plot(args.episodes, total_scores, 'Scores')
    elif args.task == 'both':
        reward_plot(args.episodes * 2, total_scores, 'Scores')
    plt.show()
if __name__=="__main__":
    time_start = time.time()
    env0 = FalseBelief_env0(args.reward)
    env1 = FalseBelief_env1(args.reward)
    # args.task = 'both'#'zero'
    # args.mode = 'test'#'train'
    # args.save_net_N = 'net_NPC_3.pth'
    # args.save_net_a = 'net_agent_3.pth'
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
