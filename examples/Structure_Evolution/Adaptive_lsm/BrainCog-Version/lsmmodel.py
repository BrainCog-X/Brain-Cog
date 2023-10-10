
from functools import partial
from torch.nn import functional as F
from torch import nn as nn
import torchvision, pprint
from copy import deepcopy
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.base.brainarea.BrainArea import BrainArea
from braincog.base.connection.CustomLinear import *
from braincog.base.learningrule.STDP import *
from braincog.base.learningrule.BCM import *
import matplotlib.pyplot as plt


@register_model
class SNN(BaseModule):
    def __init__(self,
                 liquid_size,
                 n_agent,
                 device,
                 connectivity_matrix,
                 num_classes=3,
                 step=1,
                 node_type=LIFNode,
                 encode_type='direct',
                 lsm_th=0.3,
                 fc_th=0.3,
                 lsm_tau=3,
                 fc_tau=3,
                 tw=100,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.batchsize=n_agent

        self.node_lsm=partial(node_type, **kwargs, step=step,tau=lsm_tau,threshold=lsm_th)
        self.node_fc = partial(node_type, **kwargs, step=step,tau=fc_tau,threshold=fc_th)
        self.liquid_size=liquid_size
        self.out = torch.zeros(self.batchsize, liquid_size).to(device)
        self.device=device
        self.con=[]
        self.learning_rule=[]
        self.connectivity_matrix=connectivity_matrix
        w1tmp=nn.Linear(4,liquid_size,bias=False).to(device)
        self.con.append(w1tmp)
        w2tmp=nn.Linear(liquid_size,liquid_size,bias=False).to(device)

        self.liquid_weight=w2tmp.weight.data
        
        w2tmp.weight.data=w2tmp.weight.data*self.connectivity_matrix
        self.con.append(w2tmp)
        self.learning_rule.append(BCM(self.node_lsm(), [self.con[0], self.con[1]]))  # pm
        self.fc = nn.Linear(liquid_size,num_classes).to(device)

 
        self.learning_rule.append(BCM(self.node_fc(), [self.fc]))  # pm

        

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        sum_spike=0
        time_window=20
        self.tw=time_window
        self.firing_tw=torch.zeros(time_window, self.batchsize, self.liquid_size).to(self.device)
        self.out = torch.zeros(self.batchsize, self.liquid_size).to(self.device)
        for t in range(time_window):

            self.out, self.dw = self.learning_rule[0](x, self.out)
            # self.con[0].weight+=self.dw[0]
            self.con[1].weight.data+=self.dw[1]

            out_liquid=self.out[:,0:self.liquid_size]

            xout,dw = self.learning_rule[1](out_liquid)
            self.fc.weight.data+=dw[0]
            sum_spike=sum_spike+xout
            self.firing_tw[t]=out_liquid

        outputs = sum_spike / time_window
        return outputs

