'''
Created on 2016.4.8

@author: liangqian
'''
#from modal.izhikevich import Izhikevich
from braincog.base.node import IzhNodeMU
import math
import random
import numpy as np

class IzhikevichNeuron(IzhNodeMU):
    '''
    classdocs
    '''


    def __init__(self, a = 0.1,b = 0.2,c = -65,d = 8,vthresh = 30, dt=0.1):
        '''
        Constructor
        '''
        super().__init__(threshold=vthresh, a=a, b=b, c=c, d=d, dt=dt)
        self.layerType = 'S'  # S:sequenceLayer, G: goal layer
        self.layerIndex = 0  # the layer in which the neuron situated
        self.groupIndex = 0  # the group in which the neuron situated
        self.index = 0  # starting with 1
        self.areaName = ''
        self.synapses = [] #this neuron is considered as post-synaptic neuron
        self.spiketime = []
        self.pre_neurons = []
        self.I_syn_lower = 0
        self.I_syn_upper = 0
        self.I_ext = 0
        self.I_lower = 0
        self.I_upper = 0
        self.I_ext = -100
        self.timeWindow = 5  # ms
        self.I_bg = random.randint(0, 10)
        # self.state = 'Learn' # else test
        self.selectivity = 0
        self.importance = 0
        self.preActive = False
        self.I = 0
        self.v = -65
        self.u = b * self.v
        self.vthresh = vthresh
        self.spike = False
        self.type = 'exc'
        
    def update_old(self,dt,t):
        self.spike = 0
        self.updateSynapses(t)
        self.updateCurrentOfLowerAndUpperLayer(t)
        self.I = self.I_ext + self.I_syn_lower + self.I_syn_upper
        self.v += dt * (0.04*self.v * self.v + 5 * self.v + 140 - self.u + self.I)
        self.u += dt * self.a * (self.b *self.v - self.u)
        #self.synapseWeightsDepression()
        
        if(self.v >= 30):
            self.spike = 1
            self.v = self.c
            self.u += self.d
            self.spiketime.append(t)

    def update(self,dt,t,state):
        if (state == 'Learn'):
            self.update_learn(dt, t)
        if(state == 'test'):
            self.update_test(dt,t)
    def update_learn(self,dt,t):
        self.spike = False
        self.v += dt * (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + self.I)
        self.u += dt * self.a * (self.b * self.v - self.u)
        if self.v > self.vthresh:
            self.spike = True
            self.v = self.c
            self.u += self.d
            self.preActive = True
            self.spiketime.append(t)
            self.updateSynapses(t)

    def update_test(self,dt,t):
        self.spike = False
        self.v += dt * (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + self.I)
        self.u += dt * self.a * (self.b * self.v - self.u)
        if self.v > self.vthresh:
            self.spike = True
            self.v = self.c
            self.u += self.d

    def update_normal(self,dt,t):
        self.spike = False
        self.v += dt * (0.04 * self.v * self.v + 5 * self.v + 140 - self.u + self.I)
        self.u += dt * self.a * (self.b * self.v - self.u)
        if self.v >= self.vthresh:
            self.spike = True
            self.v = self.c
            self.u += self.d
            self.spiketime.append(t)


    def updateSynapses(self,t):
        for syn in self.synapses:
            syn.computeWeight(t) 
    
    def updateCurrentOfLowerAndUpperLayer(self,t):
        I_inh = 0
        I_ext = 0
        I_exc_ext = 0
        for syn in self.synapses:
            # compute the alpha value of all spikes before this time t
            alpha_value = 0
            for st in syn.pre.spiketime:
                temp = 0
                if(t - st >= 0): temp = 6*(t/1000)*math.exp(-0.03*(t - st)/1000)
                else:temp = 0
                alpha_value += temp
            if(syn.type == 0): # from the same group
                if(syn.pre.type == 'inh'):
                    I_inh += syn.weight * (self.v+80) * alpha_value
                if(syn.pre.type == 'exc'):
                    I_ext += syn.weight * self.v * alpha_value
            if(syn.type == 1):# from other modules in the same layer
                I_exc_ext += syn.weight * self.v * alpha_value
            if(syn.type == 2): # from the upper layer
                self.I_syn_upper += self.weight * self.v * alpha_value
                
        self.I_syn_lower = -I_inh + I_ext + I_exc_ext

    def setTestStates(self):
        self.t_rest = 0
        self.spiketime = []
        self.v = -65
        self.u = self.b*self.v
        self.I = 0
        self.I_ext = 0
        for syn in self.synapses:
            syn.strength = 0

    def writeBasicInfoToJson(self):
        dic = {}
        dic["TrackID"] = self.layerIndex
        dic["GroupID"] = self.groupIndex
        dic["Index"] = self.index
        dic["selectivity"] = self.selectivity
        dic["area"] = self.areaName
        slist = []
        for syn in self.synapses:
            if (syn.weight <= 0): continue
            tmp = {}
            tmp["type"] = syn.type
            tmp["StrackID"] = syn.pre.layerIndex
            tmp["SgroupID"] = syn.pre.groupIndex
            tmp["Sindex"] = syn.pre.index
            tmp["Sarea"] = syn.pre.areaName
            tmp["pre-selectivity"] = syn.pre.selectivity
            tmp["weight"] = syn.weight
            slist.append(tmp)
        dic["synapses"] = slist
        return dic

    def writeSpikeTimeToJson(self):
        slist = []
        for i, t in enumerate(self.spiketime):
            dic = {}
            dic[i + 1] = round(t, 2)#两位有效数字
            slist.append(dic)
        return slist
        


class NoteIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh = 30):
        IzhikevichNeuron.__init__(self,a,b,c,d,vthresh)
    def setPreference(self):
        self.selectivity = self.index - 2
    def computeFilterCurrent(self):
        if(self.I_ext == self.selectivity):
            self.I = 30

    def updateCurrentOfLowerAndUpperLayer(self, t):
        self.I_lower = 0
        self.I_upper = 0
        for syn in self.synapses:
            syn.computeShortTermFacilitation2(t)
            if (syn.type == 0):  # the same group
                if (syn.excitability == 0):
                    self.I_lower -= syn.weight * syn.strength
                    if (self.I_lower < -20): self.I_lower = -20
            if (syn.type == 1):  # pre and post neurons come from  the same layer but not the same group
                #if(syn.weight > 0):
                    # print('pre_neuron_group id:'+str(syn.pre.groupIndex) + ' neuron index:'+str(syn.pre.index))
                    # print('post_neuron_group id:'+str(self.groupIndex) + ' neuron index:'+str(self.index))
                    # print('syn.strength=' + str(syn.strength))
                    # print('syn.weight='+ str(syn.weight))

                self.I_lower += syn.weight * syn.strength
                # print('syn.strength='+str(syn.strength))
                # print('syn.weight='+ str(syn.weight))

            if (syn.type >= 2):  # pre and post neurons come from the different layers
                # print(syn.pre.groupIndex)
                self.I_upper += syn.weight * syn.strength
        self.I = self.I_lower + self.I_upper
class TempoIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh = 30):
        IzhikevichNeuron.__init__(self,a,b,c,d,vthresh)
    def setPreference(self):
        self.selectivity = self.index * 0.125

    def computeFilterCurrent(self):
        if(self.I_ext <= self.selectivity + 0.0625 and self.I_ext >= self.selectivity - 0.0625 ):
            self.I = 30

    def updateCurrentOfLowerAndUpperLayer(self, t):
        self.I_lower = 0
        self.I_upper = 0
        for syn in self.synapses:
            syn.computeShortTermFacilitation2(t)
            if (syn.type == 0):  # the same group
                if (syn.excitability == 0):
                    self.I_lower -=  syn.weight * syn.strength
                    if (self.I_lower < -20): self.I_lower = -20
            if (syn.type == 1):  # pre and post neurons come from  the same layer but not the same group
                #                 if(syn.weight > 0):
                #
                #                     print('pre_neuron_group id:'+str(syn.pre.groupIndex) + ' neuron index:'+str(syn.pre.index))
                #                     print('post_neuron_group id:'+str(self.groupIndex) + ' neuron index:'+str(self.index))
                #                     print(syn.weight)
                self.I_lower += syn.weight * syn.strength
                # print('syn.strength='+str(syn.strength))

            if (syn.type == 2):  # pre and post neurons come from the different layers
                # print(syn.pre.groupIndex)
                # self.I_lower = syn.weight * syn.strength
                self.I_upper += syn.weight * syn.strength

        #         if(self.I_upper == 0):
        #             self.I = self.I_ext
        else:
            self.I = self.I_lower + self.I_upper

class TitleIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh=30):
        IzhikevichNeuron.__init__(self,a,b,c,d,vthresh)

class ComposerIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh=30):
        IzhikevichNeuron.__init__(self, a,b,c,d,vthresh)

class GenreIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self, a = 0.1,b = 0.2,c = -65,d = 8, vthresh=30):
        IzhikevichNeuron.__init__(self, a, b, c, d, vthresh)

class AmyIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh=30):
        IzhikevichNeuron.__init__(self, a,b,c,d,vthresh)

class DirectionIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh=30):
        IzhikevichNeuron.__init__(self,a,b,c,d,vthresh)

    def setPreference(self):
        # self.selectivity = 2 * math.pi/240 * self.index - math.pi/240
        self.selectivity = (self.index + 1) * math.pi / 120

    def computeFilterCurrent(self, input):
        if (input < self.selectivity + math.pi / 240 and input >= self.selectivity - math.pi / 240):
            self.I = self.I_ext = 30

    def updateCurrentOfLowerAndUpperLayer(self, t):
        self.I_lower = 0
        self.I_upper = 0
        for syn in self.synapses:
            syn.computeShortTermFacilitation2(t)
            if (syn.type == 0):  # the same group
                if (syn.excitability == 0):
                    self.I_lower -= syn.weight * syn.strength
                    if (self.I_lower < -20): self.I_lower = -20
            if (syn.type == 1):  # pre and post neurons come from  the same layer but not the same group
                #if(syn.weight > 0):In t
                    # print('pre_neuron_group id:'+str(syn.pre.groupIndex) + ' neuron index:'+str(syn.pre.index))
                    # print('post_neuron_group id:'+str(self.groupIndex) + ' neuron index:'+str(self.index))
                    # print('syn.strength=' + str(syn.strength))
                    # print('syn.weight='+ str(syn.weight))

                self.I_lower += syn.weight * syn.strength
                # print('syn.strength='+str(syn.strength))
                # print('syn.weight='+ str(syn.weight))

            if (syn.type >= 2):  # pre and post neurons come from the different layers
                # print(syn.pre.groupIndex)
                self.I_upper += syn.weight * syn.strength
        self.I = self.I_lower + self.I_upper

class GridIzhikevichCell(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh = 30):
        IzhikevichNeuron.__init__(self, a,b,c,d,vthresh)
class KeyIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh = 30):
        IzhikevichNeuron.__init__(self,a,b,c,d,vthresh)

class ModeIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8,vthresh = 30):
        IzhikevichNeuron.__init__(self,a,b,c,d,vthresh)

class ChordIzhikevichNeuron(IzhikevichNeuron):
    def __init__(self,a = 0.1,b = 0.2,c = -65,d = 8, vthresh = 30):
        IzhikevichNeuron.__init__(self, a,b,c,d,vthresh)

