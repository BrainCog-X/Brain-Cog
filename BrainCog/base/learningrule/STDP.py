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


class STDP(nn.Module):
    """
    STDP learning rule
    """

    def __init__(self, node, connection, decay=0.99):
        """
        :param node:node神经元类型实例如IFNode LIFNode
        :param connection:连接 类的实例 里面只能有一个操作
        """
        super().__init__()

        self.node = node
        self.connection = connection
        self.trace = None
        self.decay = decay

    def forward(self, x):
        """
        计算前向传播过程
        :return:s是脉冲 dw更新量
        """
        x = x.clone().detach()
        i = self.connection(x)
        with torch.no_grad():
            s = self.node(i)

            i.data += s - i.data
            trace = self.cal_trace(x)
            x.data += trace - x.data

        dw = torch.autograd.grad(outputs=i, inputs=self.connection.weight, grad_outputs=i)

        return s, dw

    def cal_trace(self, x):
        """
        计算trace
        """
        if self.trace is None:
            self.trace = Parameter(x.clone().detach(), requires_grad=False)
        else:
            self.trace *= self.decay
            self.trace += x
        return self.trace.detach()

    def reset(self):
        """
        重置
        """
        self.trace = None


class MutliInputSTDP(nn.Module):
    """
    STDP learning rule 多组神经元输入到该节点
    """

    def __init__(self, node, connection, decay=0.99):
        """
        :param node:node神经元类型实例如IFNode LIFNode
        :param connection:连接 类的实例列表 里面只能有一个操作
        """
        super().__init__()

        self.node = node
        self.connection = connection
        self.trace = [None for i in self.connection]
        self.decay = decay

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

            trace = self.cal_trace(x)
            for xi, ti in zip(x, trace):
                xi.data += ti - xi.data

        dw = torch.autograd.grad(outputs=i, inputs=[i.weight for i in self.connection], grad_outputs=i)

        return s, dw

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
        """
        重置
        """
        self.trace = [None for i in self.connection]


class LTP(MutliInputSTDP):
    """
    STDP learning rule 多组神经元输入到该节点
    """
    pass


class LTD(nn.Module):
    """
    STDP learning rule 多组神经元输入到该节点
    """

    def __init__(self, node, connection, decay=0.99):
        """
        :param node:node神经元类型实例如IFNode LIFNode
        :param connection:连接 类的实例列表 里面只能有一个操作
        """
        super().__init__()

        self.node = node
        self.connection = connection
        self.trace = None
        self.decay = decay

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

            trace = self.cal_trace(s)
            i.data += trace - i.data

        dw = torch.autograd.grad(outputs=i, inputs=[i.weight for i in self.connection], grad_outputs=i)

        return s, dw

    def cal_trace(self, x):
        """
        计算trace
        """
        if self.trace is None:
            self.trace = Parameter(torch.zeros_like(x), requires_grad=False)
        else:
            self.trace *= self.decay
        trace = self.trace.clone().detach()
        self.trace += x
        return trace

    def reset(self):
        """
        重置
        """
        self.trace = None


class FullSTDP(nn.Module):
    """
    STDP learning rule 多组神经元输入到该节点
    """

    def __init__(self, node, connection, decay=0.99, decay2=0.99):
        """
        :param node:node神经元类型实例如IFNode LIFNode
        :param connection:连接 类的实例列表 里面只能有一个操作
        """
        super().__init__()

        self.node = node
        self.connection = connection
        self.tracein = [None for i in self.connection]
        self.traceout = None
        self.decay = decay
        self.decay2 = decay2

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
            traceout = self.cal_traceout(s)
            i.data += traceout - i.data
        dw1 = torch.autograd.grad(outputs=i, inputs=[i.weight for i in self.connection], retain_graph=True,
                                  grad_outputs=i)

        with torch.no_grad():
            i.data += s - i.data

            tracein = self.cal_tracein(x)
            for xi, ti in zip(x, tracein):
                xi.data += ti - xi.data

        dw2 = torch.autograd.grad(outputs=i, inputs=[i.weight for i in self.connection], grad_outputs=i)

        return s, dw2, dw1

    def cal_tracein(self, x):
        """
        计算trace
        """
        for i in range(len(x)):
            if self.tracein[i] is None:
                self.tracein[i] = Parameter(x[i].clone().detach(), requires_grad=False)
            else:
                self.tracein[i] *= self.decay
                self.tracein[i] += x[i].detach()
        return self.tracein

    def cal_traceout(self, x):
        """
        计算trace
        """
        if self.traceout is None:
            self.traceout = Parameter(torch.zeros_like(x), requires_grad=False)
        else:
            self.traceout *= self.decay2
        trace = self.traceout.clone().detach()
        self.traceout += x
        return trace

    def reset(self):
        """
        重置
        """
        self.traceout = [None for i in self.connection]
        self.tracein = None


if __name__ == "__main__":
    node = IFNode()
    linear1 = nn.Linear(2, 2, bias=False)
    linear2 = nn.Linear(2, 2, bias=False)
    linear1.weight.data = torch.tensor([[1., 1], [1, 1]], requires_grad=True)
    linear2.weight.data = torch.tensor([[1., 1], [1, 1]], requires_grad=True)

    stdp = LTD(node, [linear1, linear2])
    for i in range(10):
        x, dw1 = stdp(torch.tensor([1.1, 1.1]), torch.tensor([1.1, 1.1]))
        print(dw1)
