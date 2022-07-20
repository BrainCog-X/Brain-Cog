import math
import random
import matplotlib
# matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from pygame.locals import *
import pandas as pd
import time

from braincog.model_zoo.base_module import BaseLinearModule, BaseModule
from braincog.base.learningrule.STDP import *
from braincog.base.brainarea.PFC import dlPFC
from utils.Encoder import *

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

class PFC_ToM(dlPFC):
    """
    SNNLinear
    """
    def __init__(self,
                 step,
                 encode_type,
                 in_features:int,
                 out_features:int,
                 bias,
                 node,
                 num_state,
                 greedy=0.8,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, in_features, out_features, bias, *args, **kwargs)
        self.encoder = PopEncoder(self.step, encode_type)
        self.encoder.device = torch.device('cpu')
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.node1 = node(threshold=0.5, tau=2.)
        self.node_name1 = node
        self.node2 = node(threshold=0.5, tau=2.)
        self.node_name2 = node
        self.num_state = num_state
        self.greedy = greedy
        self.fc = self._create_fc()
        self.c = self._rest_c()


    def _rest_c(self):
        c = torch.rand((self.out_features, self.in_features)) # eligibility trace
        return c

    def _create_fc(self):
        """
        the connection of the SNN linear
        @return: nn.Linear
        """
        fc = nn.Linear(in_features=self.in_features,
                  out_features=self.out_features, bias=self.bias)
        return fc

    def update_c(self, c, STDP, tau_c=0.2):
        """
        update the trace of eligibility
        @param c: a tensor to record eligibility
        @param STDP: the results of STDP
        @param tau_c: the parameter of trace decay
        @return: a update tensor to record eligibility
        Equation:
        delta_c = (-(c / tau_c) + STDP) * dela_t
        c = c + delta_c
        reference:<Solving the Distal Reward Problem through ...>
        """
        # delta_c = -(c / tau_c) + STDP           #dela_t = 1 ignore
        # c = c + delta_c
        c = c + tau_c * STDP
        return c

    def _call_reward(self, R, c, s, T_map):  # eligibility
        """
        R-STDP
        @param R: reward
        @param c: a tensor to record eligibility
        @param s: weight of network
        @param T_map: the mapping of the state-action pair
        @return: update weight of network
        Equation:
        delta_s = c * reward
        s = s + delta_s
        reference:<Solving the Distal Reward Problem through ...>
        """
        c[c > 0] =  c[c > 0] * R * 1
        c[c <= 0] = - c[c <= 0] * R * 1
        c = c.clamp(min=-1, max=1)
        # print('before',s[:, torch.where(T_map.gt(0))[1][0]])
        s = s + c * T_map
        # # print('after',s[:, torch.where(T_map.gt(0))[1][0]])
        s = (s - s.min(dim=0).values.unsqueeze(dim=1).T.detach().repeat(s.shape[0], 1)) / (
                s.max(dim=0).values.unsqueeze(dim=1).T.detach().repeat(s.shape[0], 1) -
                s.min(dim=0).values.unsqueeze(dim=1).T.detach().repeat(s.shape[0], 1)
        )
        # s = s * 0.5
        return s

    def update_s(self, R, mapping):
        T_map = torch.zeros((self.out_features, self.in_features))
        T_map[mapping['action']*C:mapping['action']*C+C,\
        torch.where(torch.tensor(self.encoder(mapping['state'],\
                                              self.in_features, self.num_state)[:, 0]).gt(0))]=1
        self.fc.weight.data = self._call_reward(R, self.c, self.fc.weight.data, T_map)
        # print(mapping, 'mapping')
    def forward(self, inputs, num_action, episode):
        """
        decision
        @param inputs: state
        @param num_action: num_action # consider to delete
        @return: action
        """
        inputs = self.encoder(inputs, self.in_features, self.num_state)
        count_group = torch.zeros(num_action)
        stdp = STDP(self.node2, self.fc, decay=0.80)
        # self.c = self._rest_c()
        # stdp.connection.weight.data = torch.rand((self.out_features, self.in_features))
        for t in range(self.step):
            l1_in = torch.tensor(inputs[:, t])
            l1_out = self.node1(l1_in).unsqueeze(0)  #pre  : l1_out
            l2_out, dw = stdp(l1_out)   #dw -- STDP
            self.c = self.update_c(self.c, dw[0])

        # l2_out = l2_out.T
        for i in range(num_action):
            count_group[i] = l2_out.T[i * num_depop:(i + 1) * num_depop].sum()
        # exploration or exploitation
        epsilon = random.random()
        if epsilon < self.greedy + episode * 0.004:#:
            action = count_group.argmax()
        else:
            action = torch.tensor(random.randint(0, 3))

        return action.item()





