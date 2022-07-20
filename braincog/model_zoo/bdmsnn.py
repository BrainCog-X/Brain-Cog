
import torch
from torch import nn

from braincog.base.node.node import IFNode, SimHHNode
from braincog.base.learningrule.STDP import STDP, MutliInputSTDP
from braincog.base.connection.CustomLinear import CustomLinear
from braincog.base.brainarea.basalganglia import basalganglia

import pygame
from pygame.locals import *
from collections import deque
from random import randint
#os.environ["SDL_VIDEODRIVER"] = "dummy"


class BDMSNN(nn.Module):
    def __init__(self, num_state, num_action, weight_exc, weight_inh, node_type):
        """
        定义BDM-SNN网络
        :param num_state: 状态个数
        :param num_action: 动作个数
        :param weight_exc: 兴奋性连接权重
        :param weight_inh: 抑制性连接权重
        """
        super().__init__()
        # parameters
        BG = basalganglia(num_state, num_action, weight_exc, weight_inh, node_type)
        dm_connection = BG.getweight()
        dm_mask = BG.getmask()
        # input-dlpfc
        con_matrix9 = torch.eye((num_state), dtype=torch.float)
        dm_connection.append(CustomLinear(weight_exc * con_matrix9, con_matrix9))
        dm_mask.append(con_matrix9)
        # gpi-th
        con_matrix10 = torch.eye((num_action), dtype=torch.float)
        dm_mask.append(con_matrix10)
        dm_connection.append(CustomLinear(weight_inh * con_matrix10, con_matrix10))
        # th-pm
        dm_mask.append(con_matrix10)
        dm_connection.append(CustomLinear(weight_exc * con_matrix10, con_matrix10))
        # dlpfc-th
        con_matrix11 = torch.ones((num_state, num_action), dtype=torch.float)
        dm_mask.append(con_matrix11)
        dm_connection.append(CustomLinear(0.1 * weight_exc * con_matrix11, con_matrix11))
        # pm-pm
        con_matrix3 = torch.ones((num_action, num_action), dtype=torch.float)
        con_matrix4 = torch.eye((num_action), dtype=torch.float)
        con_matrix5 = con_matrix3 - con_matrix4
        con_matrix5 = con_matrix5
        dm_mask.append(con_matrix5)
        dm_connection.append(CustomLinear(0.1 * weight_inh * con_matrix5, con_matrix5))
        # dlpfc thalamus pm +bg
        self.weight_exc = weight_exc
        self.num_subDM = 8
        self.connection = dm_connection
        self.mask = dm_mask
        self.node = BG.node
        self.node_type = node_type
        if self.node_type == "hh":
            self.node.extend([SimHHNode() for i in range(self.num_subDM - BG.num_subBG)])
            self.node[6].g_Na = torch.tensor(12)
            self.node[6].g_K = torch.tensor(3.6)
            self.node[6].g_L = torch.tensor(0.03)
        if self.node_type == "lif":
            self.node.extend([IFNode() for i in range(self.num_subDM - BG.num_subBG)])
        self.learning_rule = BG.learning_rule
        self.learning_rule.append(MutliInputSTDP(self.node[5], [self.connection[10], self.connection[12]]))  # gpi-丘脑
        self.learning_rule.append(MutliInputSTDP(self.node[6], [self.connection[11], self.connection[13]]))  # pm
        self.learning_rule.append(STDP(self.node[7], self.connection[9]))

        out_shape = [self.connection[0].weight.shape[1], self.connection[1].weight.shape[1], self.connection[2].weight.shape[1], self.connection[4].weight.shape[1], self.connection[3].weight.shape[1], self.connection[10].weight.shape[1], self.connection[11].weight.shape[1], self.connection[9].weight.shape[1]]
        self.out = []
        self.dw = []
        for i in range(self.num_subDM):
            self.out.append(torch.zeros((out_shape[i]), dtype=torch.float))
            self.dw.append(torch.zeros((out_shape[i]), dtype=torch.float))

    def forward(self, input):
        """
        根据输入得到网络的输出
        :param input: 输入
        :return: 网络的输出
        """
        self.out[7] = self.node[7](self.connection[9](input))
        self.out[0], self.dw[0] = self.learning_rule[0](self.out[7])
        self.out[1], self.dw[1] = self.learning_rule[1](self.out[7])
        self.out[2], self.dw[2] = self.learning_rule[2](self.out[7], self.out[3])
        self.out[3], self.dw[3] = self.learning_rule[3](self.out[1], self.out[2])
        self.out[4], self.dw[4] = self.learning_rule[4](self.out[0], self.out[3], self.out[2])
        self.out[5], self.dw[5] = self.learning_rule[5](self.out[4], self.out[7])
        self.out[6], self.dw[6] = self.learning_rule[6](self.out[5], self.out[6])
        br = ["StrD1", "StrD2", "STN", "Gpe", "Gpi", "thalamus", "PM", "DLPFC"]
        for i in range(self.num_subDM):
            if torch.max(self.out[i]) > 0 and self.node_type == "hh":
                self.node[i].n_reset()
            print("every areas:", br[i], self.out[i])
        return self.out[6], self.dw

    def UpdateWeight(self, i, s, num_action, dw):
        """
        更新网络中第i组连接的权重
        :param i:要更新的连接组索引
        :param s:传入状态
        :param dw:更新权重的量
        :return:
        """
        if self.node_type == "hh":
            self.connection[i].update(0.2 * self.weight_exc * dw)
            self.connection[i].weight.data[s, [s * num_action, s * num_action + 1]] /= (self.connection[i].weight.data[s, [s * num_action, s * num_action + 1]].float().max() + 1e-12)
            self.connection[i].weight.data[s, :] = self.connection[i].weight.data[s, :] * self.weight_exc
        if self.node_type == "lif":
            dw_mean = dw[s, [s * num_action, s * num_action + 1]].mean()
            dw_std = dw[s, [s * num_action, s * num_action + 1]].std()
            dw[s, [s * num_action, s * num_action + 1]] = (dw[s, [s * num_action, s * num_action + 1]] - dw_mean) / dw_std
            dw[s, :] = dw[s, :] * self.mask[i][s, :]
            self.connection[i].update(dw)
            self.connection[i].weight.data[s, [s * num_action, s * num_action + 1]] /= (self.connection[i].weight.data[s, [s * num_action, s * num_action + 1]].float().max() + 1e-12)
        if i in [0, 1, 2, 6, 7, 11, 12]:
            self.connection[i].weight.data = torch.clamp(self.connection[i].weight.data, 0, None)
        if i in [3, 4, 5, 8, 10]:
            self.connection[i].weight.data = torch.clamp(self.connection[i].weight.data, None, 0)

    def reset(self):
        """
        reset神经元或学习法则的中间量
        :return: None
        """
        for i in range(self.num_subDM):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()

    def getweight(self):
        """
        获取网络的连接(包括权值等)
        :return: 网络的连接
        """
        return self.connection
