import torch
import torch.nn as nn
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *
from utils.MyNode import *


class TIM(BaseModule):
    def __init__(self,dim=256,encode_type='direct',in_channels=16,TIM_alpha=0.5):
        super().__init__(step=1,encode_type=encode_type)
        

        #  channels may depends on the shape of input
        self.interactor = nn.Conv1d(in_channels=in_channels,out_channels=in_channels,kernel_size=5, stride=1, padding=2, bias=True)

        self.in_lif = MyNode(tau=2.0,v_threshold=0.3,layer_by_layer=False,step=1)  #spike-driven
        self.out_lif = MyNode(tau=2.0,v_threshold=0.5,layer_by_layer=False,step=1)   #spike-driven

        self.tim_alpha = TIM_alpha

    # input [T, B, H, N, C/H]
    def forward(self, x):
        self.reset()

        T, B, H, N, CoH = x.shape   

        output = [] 
        x_tim = torch.empty_like(x[0]) 

        #temporal interaction 

        for i in range(T):
            #1st step
            if i == 0 :
                x_tim = x[i]
                output.append(x_tim)
            
            #other steps
            else:
                x_tim = self.interactor(x_tim.flatten(0,1)).reshape(B,H,N,CoH).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1-self.tim_alpha)
                x_tim = self.out_lif(x_tim)
              
                
                output.append(x_tim)
            
        output = torch.stack(output) # T B H, N, C/H

        return output # T B H, N, C/H