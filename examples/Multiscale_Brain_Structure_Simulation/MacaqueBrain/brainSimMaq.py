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
from BrainCog.base.strategy.surrogate import *
from BrainCog.base.node.node import BaseNode, IFNodev1
from BrainCog.base.learningrule.STDP import STDP,MutliInputSTDP
from BrainCog.base.connection.CustomLinear import CustomLinear

from memory_profiler import profile

class MacaqueBrain(nn.Module):
    """
    Macaque Brain Simulaion
    """
    @profile
    def __init__(self,connfile='',asz=383, nsz=3, inareas=[5]):
        """
        parameters
        connfile: file definition of the models
        asz: number of areas
        nsz: number of neuron numbers
        inareas: areas which receive outside input
        """
        super().__init__()
        self.datapath = './'
        self.asz = asz
        self.nsz = nsz
        self.conn = np.zeros((asz,asz))

        self.loadConn(connfile)

        self.areas = []
        self.learn = []
        self.intype = []
        for ii in range(0,asz):
            self.areas.append([IFNodev1() for jj in range(self.nsz)])
            #self.intype.append([CustomLinear(torch.ones(1)) for jj in range(0,self.nsz)])
            #self.learn.append([STDP(self.areas[ii][jj],self.intype[ii][jj]) for jj in range(0,self.nsz)])

        self.connMat = []
        self.connlist = []
        '''
        for ii in range(0,asz):
            for jj in range(0,asz):
                if self.conn[ii][jj] > 0 or ii==jj:
                    self.connlist.append([ii,jj])
                    conMat = torch.zeros((nsz,nsz), dtype=torch.float)
                    for kk in range(nsz):
                        for ll in range(nsz):
                            rnd = np.random.rand()
                            if rnd>0.5:
                                continue
                            rnd = (rnd)*10000000
                            if ii!=jj:
                                conMat[kk][ll] = rnd*self.conn[ii][jj]
                            else:
                                conMat[kk][ll] = rnd
                    self.connMat.append(conMat)
        '''
        self.inputsa = inareas
        self.outs = torch.zeros(asz*nsz)


    def loadConn(self,connfile=''):
        """
        loading model parameters

        parameters
        connfile: file definition of the models
        """
        connfile = connfile
        retfile = open(connfile, 'r')
        for line in retfile.readlines():
            if line[0]=='#':
                continue
            res = line.split('\n')
            res = res[0].split('\r')
            res = res[0].split('\t')
            src = res[0].split('_')
            src = int(src[0])-1
            tgt = res[1].split('_')
            tgt = int(tgt[0])-1
            if src>=self.asz or tgt>=self.asz:
                continue
            self.conn[src][tgt] = float(res[2])
        retfile.close()

    def loadArea(self,areafile=''):
        pass

    def forward(self, input): #x是输入到所有接受输入神经元维度  input from DLPFC
        """
        running the model in one dt

        parameters
        input: input from the outside
        """
        # for index in range(5):
        #     x,d=self.stdp[index](x)
        #outs = []
        outs1 = np.zeros(asz*nsz)

        for ii in range(0,self.asz):
            ain = [0 for jj in range(0,self.nsz)]#self.outs[ii*self.nsz:(ii+1)*self.nsz]
            for jj in range(0,self.asz):
                if self.conn[jj][ii]>0:
                    #if self.connlist.isindex([ii,jj]):
                    if [jj,ii] in self.connlist:
                        idx = self.connlist.index([jj,ii])
                    else:
                        continue
                    mat = self.connMat[idx]
                    souts = self.outs[jj*self.nsz:(jj+1)*self.nsz]

                    #tmp = torch.mm(mat,souts)

                    for kk in range(self.nsz):
                        for ll in range(self.nsz):
                            ain[ll] += mat[kk][ll]*souts[kk]

            if ii in self.inputsa:
                for kk in range(self.nsz):
                    ain[kk] += input[kk]

            for kk in range(self.nsz):
                s = self.areas[ii][kk].forward(torch.tensor(ain[kk]))
                outs1[ii*self.nsz+kk] = s
                    #self.learn[ii][kk](self.intype[ii][kk](torch.tensor(aouts[kk])))

        for ii in range(0,self.asz):
            for jj in range(0,self.nsz):
                #self.areas[ii][jj].calc_spike()
                idx = ii*self.nsz+jj
                #tmp = [self.areas[ii][jj].mem,self.areas[ii][jj].spike]
                self.outs[idx] = outs1[idx]#self.areas[ii][jj].spike
        pass

    def UpdateWeight(self,i,dw):
        pass

    def reset(self):
        pass

    def getweight(self):
        pass


if __name__=="__main__":
    #print("hello")

    t = 10
    nsz = 10
    asz = 30

    mb = MacaqueBrain('./conn/macaque/edges.cfg',asz=asz,nsz=nsz)
    #from torchstat import stat
    #stat(mb)
    #exit(0)
    for ii in range(0,t):
        input = [np.random.rand()*100 for jj in range(0,nsz)]
        mb.forward(input)
        print(ii)
        print(mb.outs)
