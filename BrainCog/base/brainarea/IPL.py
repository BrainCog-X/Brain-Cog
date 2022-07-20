
from braincog.base.learningrule.STDP import *
from braincog.base.node.node import *
from braincog.base.connection.CustomLinear import *
import random
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
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from braincog.base.strategy.surrogate import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class IPLNet(nn.Module):
    """
    inferior parietal lobule (IPL)
    """

    def __init__(self, connection):
        """
        Setting the network structure of IPL
        """
        super().__init__()
        # IPLM, IPLV
        self.num_subMB = 2
        self.node = [IzhNodeMU(threshold=30., a=0.02, b=0.2, c=-65., d=6., mem=-70.) for i in range(self.num_subMB)]

        self.connection = connection
        self.learning_rule = []

        self.learning_rule.append(STDP(self.node[0], self.connection[0]))  # vPMC_input-IPLM
        self.learning_rule.append(MutliInputSTDP(self.node[1], [self.connection[1], self.connection[2]]))  # STS_input-IPLV, IPLM-IPLV

        self.out_IPLM = torch.zeros((self.connection[0].weight.shape[1]), dtype=torch.float)
        self.out_IPLV = torch.zeros((self.connection[1].weight.shape[1]), dtype=torch.float)

    def forward(self, input1, input2):  # input from vPMC and STS
        """
        Calculate the output of IPLv and the weight update between IPLm and IPLv
        :param input1: input from vPMC
        :param input2: input from STS
        :return: output of IPLv, weight update between IPLm and IPLv
        """
        self.out_IPLM = self.node[0](self.connection[0](input1))
        self.out_IPLV, dw_IPLv = self.learning_rule[1](input2, self.out_IPLM)
        if sum(sum(self.out_IPLV)) == 1:
            dw_IPLv = dw_IPLv[0][torch.nonzero(dw_IPLv[1])[0][1]][torch.nonzero(dw_IPLv[1])[0][1]] * dw_IPLv[1]
        else:
            dw_IPLv = dw_IPLv[0]
        return self.out_IPLV, dw_IPLv

    def UpdateWeight(self, i, dw):
        """
        Update the weight
        :param i: index of the connection to update
        :param dw: weight update
        :return: None
        """
        self.connection[i].update(dw)

    def reset(self):
        """
        reset the network
        :return: None
        """
        for i in range(self.num_subMB):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()

    def getweight(self):
        """
        Get the connection and weight in IPL
        :return: connection
        """
        return self.connection
