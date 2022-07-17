'''
Primary Auditory Cortex
'''
import torch
from BrainCog.base.node.node import *
from BrainCog.base.brainarea.BrainArea import *
from BrainCog.base.connection import CustomLinear
from BrainCog.base.learningrule.STDP import *

class PAC(BrainArea):

    def __int__(self,w,mask):
        self.noteNetworks = NoteLIFNode()
        self.connection = [CustomLinear(w,mask),CustomLinear(w2,mask2)]
        self.stdp = []
        self.internalinputs = torch.zeros(640,640)
        self.stdp.append(MutliInputSTDP(self.noteNetworks, self.connection))

    def forward(self, x):
        self.internalinputs,dw = self.stdp[0](x,self.internalinputs)
        return self.internalinputs, dw