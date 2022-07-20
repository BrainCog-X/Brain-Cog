import numpy as np
import torch, os, sys
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
from braincog.base.node.node import *
from braincog.base.learningrule.STDP import *
from braincog.base.connection.CustomLinear import *


class BrainArea(nn.Module, abc.ABC):
    """
    脑区基类
    """

    @abc.abstractmethod
    def __init__(self):
        """
        """
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        """
        计算前向传播过程
        :return:x是脉冲
        """

        return x

    def reset(self):
        """
        计算前向传播过程
        :return:x是脉冲
        """

        pass


class ThreePointForward(BrainArea):
    """
    三点前馈脑区
    """

    def __init__(self, w1, w2, w3):
        """
        """
        super().__init__()

        self.node = [IFNode(), IFNode(), IFNode()]
        self.connection = [CustomLinear(w1), CustomLinear(w2), CustomLinear(w3)]
        self.stdp = []

        self.stdp.append(STDP(self.node[0], self.connection[0]))
        self.stdp.append(STDP(self.node[1], self.connection[1]))
        self.stdp.append(STDP(self.node[2], self.connection[2]))

    def forward(self, x):
        """
        计算前向传播过程
        :return:x是脉冲
        """
        x, dw1 = self.stdp[0](x)
        x, dw2 = self.stdp[1](x)
        x, dw3 = self.stdp[2](x)

        return x, (*dw1, *dw2, *dw3)


class Feedback(BrainArea):
    """
    反馈网络
    """

    def __init__(self, w1, w2, w3):
        """
        """
        super().__init__()

        self.node = [IFNode(), IFNode()]
        self.connection = [CustomLinear(w1), CustomLinear(w2), CustomLinear(w3)]
        self.stdp = []

        self.stdp.append(MutliInputSTDP(self.node[0], [self.connection[0], self.connection[2]]))
        self.stdp.append(STDP(self.node[1], self.connection[1]))
        self.x1 = torch.zeros(1, w3.shape[0])

    def forward(self, x):
        """
        计算前向传播过程
        :return:x是脉冲
        """
        x, dw1 = self.stdp[0](x, self.x1)
        self.x1, dw2 = self.stdp[1](x)

        return self.x1, (*dw1, *dw2)

    def reset(self):
        self.x1 *= 0


class TwoInOneOut(BrainArea):
    """
    反馈网络
    """

    def __init__(self, w1, w2):
        """
        """
        super().__init__()

        self.node = [IFNode()]
        self.connection = [CustomLinear(w1), CustomLinear(w2)]
        self.stdp = []

        self.stdp.append(MutliInputSTDP(self.node[0], [self.connection[0], self.connection[1]]))

    def forward(self, x1, x2):
        """
        计算前向传播过程
        :return:x是脉冲
        """
        x, dw1 = self.stdp[0](x1, x2)

        return x, dw1


class SelfConnectionArea(BrainArea):
    """
    反馈网络
    """

    def __init__(self, w1, w2 ):
        """
        """
        super().__init__()

        self.node = [IFNode() ]
        self.connection = [CustomLinear(w1), CustomLinear(w2) ]
        self.stdp = []

        self.stdp.append(MutliInputSTDP(self.node[0], [self.connection[0], self.connection[1]]))
        self.x1 = torch.zeros(1, w2.shape[0])

    def forward(self, x):
        """
        计算前向传播过程
        :return:x是脉冲
        """
        self.x1, dw1 = self.stdp[0](x, self.x1)


        return self.x1, dw1

    def reset(self):

        self.x1 *= 0

if __name__ == "__main__":
    T = 20
    w1 = torch.tensor([[1., 1], [1, 1]])
    w2 = torch.tensor([[1., 1], [1, 1]])
    w3 = torch.tensor([[0.4, 0.4], [0.4, 0.4]])
    ba = TwoInOneOut(w1, w2)
    for i in range(T):
        x = ba(torch.tensor([[0.1, 0.1]]), torch.tensor([[0.1, 0.1]]))
        print(x[0])
