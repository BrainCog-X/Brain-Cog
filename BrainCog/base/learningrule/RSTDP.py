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
from braincog.base.node import *


class RSTDP(nn.Module):
    """
    RSTDP算法
    """
    def __init__(self, node, connection, decay=0.99, reward_decay=0.5):
        """
        :param node:node神经元类型实例如IFNode LIFNode
        :param connection:连接 类的实例列表 里面只能有一个操作
        """
        super().__init__()

        self.node = node
        self.connection = connection
        if not isinstance(connection, list):
            self.connection = [self.connection]
        self.trace = [None for i in self.connection]
        self.decay = decay
        self.reward_decay = reward_decay
        self.stdp = STDP(self.node, self.node, self.decay)

    def forward(self, *x, r):
        """
        计算前向传播过程
        :return:s是脉冲 dw更新量
        """
        s, dw = self.stdp(x)
        trace = self.cal_trace(r)
        return s, dw * trace

    def cal_trace(self, x):
        """
        计算trace
        """
        for i in range(len(x)):
            if self.trace[i] is None:
                self.trace[i] = Parameter(x[i].clone().detach(), requires_grad=False)
            else:
                self.trace[i] *= self.decay
                self.trace[i] += x[i].detach()
        return self.trace

    def reset(self):
        self.trace = [None for i in self.connection]
