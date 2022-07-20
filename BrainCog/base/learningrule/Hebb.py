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
from braincog.base.node.node import *


class Hebb(nn.Module):
    """
    Hebb learning rule 多组神经元输入到该节点
    """

    def __init__(self, node, connection):
        """
        :param node:node神经元类型实例如IFNode LIFNode
        :param connection:连接 类的实例列表 里面只能有一个操作
        """
        super().__init__()

        self.node = node
        self.connection = connection
        self.trace = [None for i in self.connection]

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

            i.data += s - i.data

        dw = torch.autograd.grad(outputs=i, inputs=[i.weight for i in self.connection], grad_outputs=i)

        return s, dw

    def reset(self):
        """
        重置
        """
        self.trace = [None for i in self.connection]


if __name__ == "__main__":
    node = IFNode()
    linear1 = nn.Linear(2, 2, bias=False)
    linear2 = nn.Linear(2, 2, bias=False)
    linear1.weight.data = torch.tensor([[1., 1], [1, 1]], requires_grad=True)
    linear2.weight.data = torch.tensor([[1., 1], [1, 1]], requires_grad=True)

    hebb = Hebb(node, [linear1, linear2])
    for i in range(10):
        x, dw1 = hebb(torch.tensor([1.1, 1.1]), torch.tensor([1.1, 1.1]))
        print(dw1)
