import torch
from braincog.base.brainarea.Insula import *
from rulebasedpolicy.world_model import *
from BrainArea.dACC import *
from BrainArea.PFC_ToM import *

NPC_1 = 2
NPC_2 = 3
Agent = 4

#exploit or explore
num_enpop = 6
num_depop = 10
greedy = 0.8#0.5

#state
A_state = 4
N_state = 6
cell_num = 6
#action
C=10

class ToM:
    def __init__(self, env):
        """

        @param axis:输入为agent自己的观察到位置信息
        @param obs:遮挡关系
        """
        self.axis = None
        self.obs = None
        self.NPC_num = None
        self.env = env
        self.env.trigger = 0

    def TPJ(self, NPC_num, axis, obs):
        """
        perspective_taking
        agent take NPC2's perspective
        @param NPC_num: which NPC?
        @return:
        axis_new:站在other的角度看到其他智能体的遮挡关系,return axis,
        axis_switch:站在self的角度看到其他智能体的遮挡关系,return axis
        obs_switch:站在other的角度看到其他智能体的遮挡关系,return obs
        """
        self.env.trigger = 0
        axis_switch = [[6,6], [6,6], [6,6]]
        axis_new = [[6, 6], [6, 6], [6, 6]]
        self.axis = axis
        self.obs = obs
        axis_switch[0], axis_switch[NPC_num] = axis[NPC_num], axis[0]
        axis_switch[1] = axis[1]
        obs_switch = big_env(self.obs)
        obs_switch[self.axis[0][0], self.axis[0][1]], obs_switch[self.axis[NPC_num][0],self.axis[NPC_num][1]] = \
            obs_switch[self.axis[NPC_num][0],self.axis[NPC_num][1]],obs_switch[self.axis[0][0], self.axis[0][1]]
        x = np.argwhere((obs_switch==2)|(obs_switch==8))
        if self.axis[NPC_num][0] != 6 or self.axis[NPC_num][1] != 6:
            shelter_obs = shelter_env(obs_switch[1:6,1:6])
            obs_switch[1:6,1:6], m = self.gain_obs(a=obs_switch[1:6,1:6], aa=shelter_obs, b=axis_switch[1], c=axis_switch[2], bb=2,cc=4)
            if m == True:
                axis_switch[1] = [6,6]
        else:
            obs_switch = []
        axis_new[0] = axis_switch[NPC_num]
        axis_new[1] = axis_switch[1]
        axis_new[NPC_num] = axis_switch[0]

        return  axis_new, axis_switch, obs_switch

    def gain_obs(self, a,aa,b,c,bb,cc):
        m = False
        if b!=[6,6]:
            if aa[b[0]-1, b[1]-1] == 0:
                a[b[0]-1, b[1]-1] = 1#2
                m = True
            else:
                a[b[0] - 1, b[1] - 1] = bb
        if aa[c[0]-1, c[1]-1] ==0:
            # print('-------')
            a[c[0]-1, c[1]-1] = 1#4
        else:
            a[c[0] - 1, c[1] - 1] = cc
        return a, m

    def belief_reasoning(self, test_x, net_NPC, num_action, episode):
        output = net_NPC(inputs=test_x, \
                              num_action=num_action, \
                              episode=episode)
        return output

    def state_evaluation(self, prediction_next_state):
        """
        state_evaluation
        @param prediction_next_state:
        @return:
        """
        input = np.array(prediction_next_state)
        test_x = torch.tensor([[(int(bool(input[0][0] - input[2][0])))*10, (int(bool(input[0][1] - input[2][1])))*10]])
        T = 5
        num_popneurons = 2
        safety = 2
        dACC_net = dACC(step=T, encode_type='rate', bias=True,
                        in_features=num_popneurons, out_features=safety,
                        node=node.LIFNode)
        dACC_net.load_state_dict(torch.load(os.path.join(sys.path[0], 'BrainArea/checkpoint', 'dACC_net.pth'))['dacc'])
        output = dACC_net(inputs=test_x, epoch=50)
        output = bool(int(output[0].cpu().detach().numpy().tolist()))
        print(output,test_x)
        return output

    def prediction_state(self, axis_new, axis, action_NPC1, net, num_action, episode):
        """
        根据当前状态和经验预测下一个状态
        @return:下一个step的state
        """
        self.env.trigger = 0
        action_move = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0),
            4: (0, 0)
        }
        next_axis = [[6,6],[6,6],[6,6]]
        # inputspike_test = np.array([axis_new[0],axis_new[1],axis_new[2]])
        inputspike_test = sum(axis_new, [])

        action_NPC2 = self.belief_reasoning(test_x=inputspike_test, net_NPC=net, num_action=num_action, episode=episode)
        action_agent = 3
        #NPC_1
        next_axis[1][0] = axis[1][0] + action_move[action_NPC1][1]
        next_axis[1][1] = axis[1][1] + action_move[action_NPC1][0]
        #NPC_2

        if self.obs[axis[2][0] + action_move[action_NPC2][1]-1, axis[2][1] + action_move[action_NPC2][0]-1] != 5:
            next_axis[2][0] = axis[2][0] + action_move[action_NPC2][1]
            next_axis[2][1] = axis[2][1] + action_move[action_NPC2][0]
        #NPC_agent
        next_axis[0][0] = axis[0][0] + action_move[action_agent][1]
        next_axis[0][1] = axis[0][1] + action_move[action_agent][0]

        return next_axis

    def altruism(self, axis_switch, axis_NPC, n_actions):
        """
        假设有一个开关，agent按下去可以让NPC不动
        Q_bad:NPC的错误观测的有偏差Q
        Q_good:正确的Q
        Q_delta:中最小的值就是容易导致NPC出现危险的值
        找到最小危险中的最大值对应的action
        @param axis_switch:
        @param axis_NPC:
        @param n_actions:
        @return:下一个step的action
        """
        actions = list(range(n_actions))
        action_NPC_list = list(range(n_actions))
        #others' view
        data_NPC = pd.read_csv('./data/NPC_assessment.csv', index_col=[0],
                               dtype={1: np.float64, 2: np.float64, 3: np.float64, 4: np.float64,
                                      5: np.float64})
        #self's view
        data_agent = pd.read_csv('./data/agent_assessment.csv', index_col=[0],
                               dtype={1: np.float64, 2: np.float64, 3: np.float64, 4: np.float64,
                                      5: np.float64})

        # print(axis_NPC, axis_switch)
        Q_bad = data_NPC.loc[str(axis_NPC), :]
        if str(axis_switch) not in data_agent.index:
            # append new state to q table
            # print('1')
            data_agent = data_agent.append(
                pd.Series(
                    [0] * len(list(range(self.env.n_actions))),
                    index=data_agent.columns,
                    name=str(axis_switch),
                    )
            )
        Q_good = data_agent.loc[str(axis_switch), :]
        Q_delta = Q_good - Q_bad
        # print(Q_delta)
        # max_Q_delta = [None] * n_actions
        min_Q_delta_set = []
        #stop
        for action_a in actions:
            if action_a == 4:
                action_NPC_list = [4]

            min_Q_delta = []
            for i in action_NPC_list:
                #
                # print(i)
                min_Q_delta.append(Q_delta[i])

            min_Q_delta_set.append(min(min_Q_delta))
        # print(min_Q_delta_set,'---------')
        action_altruism = min_Q_delta_set.index(max(min_Q_delta_set))

        if action_altruism  == 4:
            self.env.trigger = 1
            # print('---------------------------------------------')
        # env.SHOW()
        # time.sleep(1.0)
        # max_Q_delta[action_a] = max(min_Q_delta_set)
        return action_altruism

    def INS(self, axis1, axis2):

        num_IPLM = axis1.shape[1]
        num_IPLV = axis1.shape[1]
        Insula_connection = []
        # IPLV-Insula
        con_matrix0 = torch.eye(num_IPLM, dtype=torch.float) * 2
        Insula_connection.append(CustomLinear(con_matrix0))
        # STS-Insula
        con_matrix1 = torch.eye(num_IPLV, dtype=torch.float) * 2
        Insula_connection.append(CustomLinear(con_matrix1))

        Insula = InsulaNet(Insula_connection)

        confidence = 0
        Insula.reset()
        for t in range(2):
            Insula((axis1-axis2) * 10, torch.zeros_like(axis1) * 10)
        if sum(sum(Insula.out_Insula)) > 0:
            confidence = confidence + 1

        return confidence

