
import torch
from torch import nn

from braincog.base.node.node import IFNode
from braincog.base.learningrule.STDP import STDP,MutliInputSTDP
from braincog.base.connection.CustomLinear import CustomLinear


from collections import deque
from random import randint

class RSNN(nn.Module):
    def __init__(self,num_state,num_action):
        super().__init__()
        # parameters
        rsnn_mask=[]
        rsnn_con=[]
        con_matrix1 = torch.ones((num_state,num_action), dtype=torch.float)
        rsnn_mask.append(con_matrix1)
        rsnn_con.append(CustomLinear(torch.randn(num_state,num_action), con_matrix1))

        self.num_subR=2
        self.connection = rsnn_con
        self.mask=rsnn_mask
        self.node = [IFNode() for i in range(self.num_subR)]
        self.learning_rule = []
        self.learning_rule.append(MutliInputSTDP(self.node[1], [self.connection[0]]))

        self.weight_trace = torch.zeros(con_matrix1.shape, dtype=torch.float)
        
        self.out_in = torch.zeros((num_state), dtype=torch.float)
        self.out = torch.zeros((self.connection[0].weight.size()[1]), dtype=torch.float)
        self.dw = torch.zeros((self.connection[0].weight.size()), dtype=torch.float)

    def forward(self, input):
        input=torch.tensor(input, dtype=torch.float)
        self.out_in=self.node[0](input)
        self.out,self.dw = self.learning_rule[0](self.out_in)
        return self.out,self.dw

    def UpdateWeight(self,reward):
        self.weight_trace[self.weight_trace>0]=self.weight_trace[self.weight_trace>0]*reward
        self.weight_trace[self.weight_trace < 0] = -1*self.weight_trace[self.weight_trace < 0] * reward
        self.connection[0].update(self.weight_trace)
        for i in range(self.connection[0].weight.size()[1]):
            self.connection[0].weight.data[:, i] = (self.connection[0].weight.data[:, i] - torch.min(self.connection[0].weight.data[:, i])) / (torch.max(self.connection[0].weight.data[:, i]) - torch.min(self.connection[0].weight.data[:, i]))
        self.connection[0].weight.data= self.connection[0].weight.data * 0.5
    def reset(self):
        for i in range(self.num_subR):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()
    def getweight(self):
        return self.connection
