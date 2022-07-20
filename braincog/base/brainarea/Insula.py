import numpy as np
import torch,os,sys
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
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random

from braincog.base.connection.CustomLinear import *
from braincog.base.node.node import *
from braincog.base.learningrule.STDP import *


class InsulaNet(nn.Module):
    """
    Insula
    """
    def __init__(self,connection):
        """
        Setting the network structure of Insula
        """
        super().__init__()
        # Insula
        self.num_subMB = 1
        self.node = [IzhNodeMU(threshold=30., a=0.02, b=0.2, c=-65., d=6., mem=-70.) for i in range(self.num_subMB)]
        self.connection = connection
        self.learning_rule = []        
        self.learning_rule.append(MutliInputSTDP(self.node[0], [self.connection[0],self.connection[1]]))# IPLv-Insula, STS-Insula
        self.Insula=torch.zeros((self.connection[1].weight.shape[1]), dtype=torch.float)

    def forward(self, input1, input2): # input from IPLv and STS
        """
        Calculate the output of Insula 
        :param input1: input from IPLv
        :param input2: input from STS
        :return: output of Insula, weight update (unused)
        """
        self.out_Insula, dw_Insula = self.learning_rule[0](input1, input2)
        return self.out_Insula

    def UpdateWeight(self,i,dw):
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
        Get the connection and weight in Insula
        :return: connection
        """
        return self.connection