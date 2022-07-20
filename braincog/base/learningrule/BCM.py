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


class BCM(nn.Module):
    """
    BCM learning rule 多组神经元输入到该节点
    """

    def __init__(self, node, connection, cfunc=None, weightdecay=0.99, tau=10):
        """
        :param node:node神经元类型实例如IFNode LIFNode
        :param connection:连接 类的实例列表 里面只能有一个操作
        :param cfunc:BCM的频率函数 默认y(y-th)
        :param weightdecay:权重衰减系数 默认0.99
        :param tau: 频率更新时间常数
        """
        super().__init__()

        self.node = node
        self.connection = connection
        if not isinstance(connection, list):
            self.connection = [self.connection]
        self.weightdecay = weightdecay
        self.tau = tau
        self.threshold = 0

    def forward(self, *x):
        """
        计算前向传播过程
        :return:s是脉冲 dw更新量
        """
        i = 0
        x = [xi.clone().detach() for xi in x]
        for xi, coni in zip(x, self.connection):
            i += coni(xi)
        with torch.no_grad():
            s = self.node(i)

            i.data += self.cfunc(s) - i.data

        dw = torch.autograd.grad(outputs=i, inputs=[i.weight for i in self.connection], grad_outputs=i)
        for dwi, i in zip(dw, self.connection):
            dwi -= (1 - self.weightdecay) * i.weight
        return s, dw

    def cfunc(self, s):
        self.threshold = ((self.tau - 1) * self.threshold + s) / self.tau

        return (s * (s - self.threshold)).detach()

    def reset(self):
        """
        重置
        """
        self.threshold = 0
        pass
