import numpy as np
import torch
import os
import sys
from torch import nn
from torch.nn import Parameter

import abc
import math
from abc import ABC

import numpy as np
import torch
import torch.nn.functional as F
from braincog.base.strategy.surrogate import *
from braincog.base.node.node import IFNode, SimHHNode
from braincog.base.learningrule.STDP import STDP, MutliInputSTDP
from braincog.base.connection.CustomLinear import CustomLinear


class basalganglia(nn.Module):
    """
    Basal Ganglia
    """

    def __init__(self, ns, na, we, wi, node_type):
        super().__init__()
        """
        :param ns: 状态个数
        :param na:动作个数
        :param we:兴奋性连接权重
        :param wi:抑制性连接权重
        """
        num_state = ns
        num_action = na
        num_STN = 2
        weight_exc = we
        weight_inh = wi
        # connetions: 0DLPFC-StrD1 1DLPFC-StrD2 2DLPFC-STN 3StrD1-GPi 4StrD2-GPe 5Gpe-Gpi 6STN-Gpi 7STN-Gpe 8Gpe-STN
        bg_connection = []
        bg_con_mask = []
        # DLPFC-StrD1
        con_matrix1 = torch.zeros((num_state, num_state * num_action), dtype=torch.float)
        for i in range(num_state):
            for j in range(num_action):
                con_matrix1[i, i * num_action + j] = 1
        bg_con_mask.append(con_matrix1)
        bg_connection.append(CustomLinear(weight_exc * con_matrix1, con_matrix1))
        # DLPFC-StrD2
        bg_connection.append(CustomLinear(weight_exc * con_matrix1, con_matrix1))
        bg_con_mask.append(con_matrix1)
        # DLPFC-STN
        con_matrix3 = torch.ones((num_state, num_STN), dtype=torch.float)
        bg_con_mask.append(con_matrix3)
        bg_connection.append(CustomLinear(weight_exc * con_matrix3, con_matrix3))
        # StrD1-GPi
        con_matrix4 = torch.zeros((num_state * num_action, num_action), dtype=torch.float)
        for i in range(num_state):
            for j in range(num_action):
                con_matrix4[i * num_action + j, j] = 1
        bg_con_mask.append(con_matrix4)
        bg_connection.append(CustomLinear(weight_inh * con_matrix4, con_matrix4))
        # StrD2-GPe
        bg_con_mask.append(con_matrix4)
        bg_connection.append(CustomLinear(weight_inh * con_matrix4, con_matrix4))
        # Gpe-Gpi
        con_matrix5 = torch.eye((num_action), dtype=torch.float)
        bg_con_mask.append(con_matrix5)
        bg_connection.append(CustomLinear(weight_inh * con_matrix5, con_matrix5))
        # STN-Gpi
        con_matrix6 = torch.ones((num_STN, num_action), dtype=torch.float)
        bg_con_mask.append(con_matrix6)
        bg_connection.append(CustomLinear(0.5 * weight_exc * con_matrix6, con_matrix6))
        # STN-Gpe
        bg_con_mask.append(con_matrix6)
        bg_connection.append(CustomLinear(0.5 * weight_exc * con_matrix6, con_matrix6))
        # Gpe-STN
        con_matrix7 = torch.ones((num_action, num_STN), dtype=torch.float)
        bg_con_mask.append(con_matrix7)
        bg_connection.append(CustomLinear(0.5 * weight_inh * con_matrix7, con_matrix7))

        self.num_subBG = 5
        self.node_type = node_type
        if self.node_type == "hh":
            self.node = [SimHHNode() for i in range(self.num_subBG)]
        if self.node_type == "lif":
            self.node = [IFNode() for i in range(self.num_subBG)]
        self.connection = bg_connection
        self.mask = bg_con_mask
        self.learning_rule = []

        trace_stdp = 0.99
        self.learning_rule.append(STDP(self.node[0], self.connection[0], trace_stdp))  # DLPFC-StrD1
        self.learning_rule.append(STDP(self.node[1], self.connection[1], trace_stdp))  # DLPFC-StrD2
        self.learning_rule.append(MutliInputSTDP(self.node[2], [self.connection[2], self.connection[8]]))  # DLPFC-STN
        self.learning_rule.append(MutliInputSTDP(self.node[3], [self.connection[4], self.connection[7]]))  # StrD2-GPe STN-Gpe
        self.learning_rule.append(MutliInputSTDP(self.node[4], [self.connection[3], self.connection[5], self.connection[6]]))  # StrD1-GPi Gpe-Gpi STN-Gpi
        self.out_StrD1 = torch.zeros((self.connection[0].weight.shape[1]), dtype=torch.float)
        self.out_StrD2 = torch.zeros((self.connection[1].weight.shape[1]), dtype=torch.float)
        self.out_STN = torch.zeros((self.connection[2].weight.shape[1]), dtype=torch.float)
        self.out_Gpi = torch.zeros((self.connection[3].weight.shape[1]), dtype=torch.float)
        self.out_Gpe = torch.zeros((self.connection[4].weight.shape[1]), dtype=torch.float)

    def forward(self, input):
        """
        计算由当前输入基底节网络的输出
        :param input: 输入电流
        :return: 输出脉冲
        """
        self.out_StrD1, dw_strd1 = self.learning_rule[0](input)
        self.out_StrD2, dw_strd2 = self.learning_rule[1](input)
        self.out_STN, dw_stn = self.learning_rule[2](input, self.out_Gpe)
        self.out_Gpe, dw_gpe = self.learning_rule[3](self.out_StrD2, self.out_STN)
        self.out_Gpi, dw_gpi = self.learning_rule[4](self.out_StrD1, self.out_Gpe, self.out_STN)
        return self.out_Gpi

    def UpdateWeight(self, i, dw):
        """
        更新基底节内第i组连接的权重 根据传入的dw值
        :param i: 要更新的连接的索引
        :param dw: 更新的量
        :return: None
        """
        self.connection[i].update(dw)
        self.connection[i].weight.data = F.normalize(self.connection[i].weight.data.float(), p=1, dim=1)

    def reset(self):
        """
        reset神经元或学习法则的中间量
        :return: None
        """
        for i in range(self.num_subMB):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()

    def getweight(self):
        """
        获取基底节网络的连接(包括权值等)
        :return: 基底节网络的连接
        """
        return self.connection

    def getmask(self):
        """
        获取基底节网络的连接（仅连接矩阵）
        :return: 基底节网络的连接矩阵
        """
        return self.mask


if __name__ == "__main__":
    BG = basalganglia(4, 2, 0.2, -4)
    con = BG.getweight()
    print(con)
