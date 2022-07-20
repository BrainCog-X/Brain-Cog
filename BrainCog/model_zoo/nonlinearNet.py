import torch.nn.functional as F

from braincog.base.strategy.surrogate import *
from braincog.base.node.node import IFNode
from braincog.base.learningrule.STDP import STDP, MutliInputSTDP


class droDMTestNet(nn.Module):
    """
    Drosophila Testing Network: compound eye-KC-MBON  DA-GABA-MB
    """

    def __init__(self, connection):
        """
        根据传入的连接 构建测试网络
        :param connection: 测试网络的连接
        """
        super().__init__()
        trace_stdp = 0.99
        self.num_subMB = 5
        self.node = [IFNode() for i in range(self.num_subMB)]
        self.connection = connection
        self.learning_rule = []
        self.learning_rule.append(STDP(self.node[0], self.connection[0], trace_stdp))
        self.learning_rule.append(MutliInputSTDP(self.node[1], [self.connection[1], self.connection[5]], trace_stdp))
        self.learning_rule.append(MutliInputSTDP(self.node[2], [self.connection[2], self.connection[3], self.connection[9]], trace_stdp))
        self.learning_rule.append(MutliInputSTDP(self.node[3], [self.connection[4], self.connection[6]], trace_stdp))
        self.learning_rule.append(MutliInputSTDP(self.node[4], [self.connection[7], self.connection[8]], trace_stdp))

        self.out_vis = torch.zeros((self.connection[0].weight.shape[1]), dtype=torch.float)
        self.out_KC = torch.zeros((self.connection[1].weight.shape[1]), dtype=torch.float)
        self.out_MBON = torch.zeros((self.connection[2].weight.shape[1]), dtype=torch.float)
        self.out_APL = torch.zeros((self.connection[4].weight.shape[1]), dtype=torch.float)
        self.out_DA = torch.zeros((self.connection[7].weight.shape[1]), dtype=torch.float)

    def forward(self, input, input_da):
        """
        根据输入得到输出
        :param input: 输入电流
        :return: 网络的输出，以及网络运行产生的STDP可塑性
        """
        self.out_vis = self.node[0](self.connection[0](input))
        self.out_KC, dw_kc = self.learning_rule[1](self.out_vis, self.out_APL)
        self.out_MBON, dw_mbon = self.learning_rule[2](self.out_KC, self.out_MBON, self.out_DA)
        self.out_APL, dw_apl = self.learning_rule[3](self.out_KC, self.out_DA)
        self.out_DA, dw_da = self.learning_rule[4](self.out_APL, input_da)
        return self.out_MBON, dw_kc[1], dw_apl[0]

    def UpdateWeight(self, i, dw):
        """
        更新网络中第i组连接的权重
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
        获取网络的连接(包括权值等)
        :return: 网络的连接
        """
        return self.connection
