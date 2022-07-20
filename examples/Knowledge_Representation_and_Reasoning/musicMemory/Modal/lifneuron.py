import torch
import random
from braincog.base.node import node
import numpy as np
import pylab as pl
class LIFNeuron(node.LIFNode):

    def __init__(self, tau_ref = 0, vthresh = 5, Rm = 2, Cm = 0.2,dt = 0.1,*args, **kwargs):
        super().__init__(threshold=vthresh, tau=Rm*Cm, dt=dt, *args, **kwargs)
        self.layerType = 'S'  # S:sequenceLayer, G: goal layer
        self.layerIndex = 0  # the layer in which the neuron situated
        self.groupIndex = 0  # the group in which the neuron situated
        self.index = 0  # starting with 1
        self.areaName = ''
        self.pre_neurons = []
        self.synapses = []
        self.spiketime = []
        self.type = 'exc'
        self.tau_ref = tau_ref
        self.tau_m = Rm*Cm
        self.vth = vthresh
        self.Rm = Rm
        self.Cm = Cm
        self.t_rest = 0
        self.I = 0
        self.spike = False
        self.firingrate = 0  # Hz
        self.I_ower = 0
        self.I_upper = 0
        self.I_ext = -100
        self.timeWindow = 5  # ms
        self.I_bg = random.randint(0, 10)
        # self.state = 'Learn' # else test
        self.selectivity = 0
        self.preActive = False

    def update(self, dt, t, state):  # state = 'learn' or  state = 'test'

        if (state == 'Learn'):
            self.update_learn(dt, t)
            '''
            #------------Gaussian selectivity--------------#
            self.I = math.exp(-((self.I_ext-math.pi/16)/0.24)**2)
            self.I = self.I if self.I >= 0.5 else 0
            self.I *= 10
            '''

        elif (state == 'test'):
            self.update_test(dt, t)

    def update_learn(self, dt, t):
        self.spike = False
        # self.computeFilterCurrent()
        '''
        #------------Gaussian selectivity--------------#
        self.I = math.exp(-((self.I_ext-math.pi/16)/0.24)**2)
        self.I = self.I if self.I >= 0.5 else 0
        self.I *= 10
        '''
        if (t >= self.t_rest):
            self.mem += dt * (-self.mem + self.I * self.Rm) / self.tau_m
            if (self.mem > self.vth):
                self.spike = True
                self.preActive = True
                # print("groupID:"+ str(self.groupIndex) + ", neuronID:"+str(self.index))
                self.spiketime.append(t)
                self.mem = 0
                self.t_rest = t + self.tau_ref
                self.updateSynapses(t)

    def update_test(self, dt, t):
        self.spike = False
        # self.updateCurrentOfLowerAndUpperLayer(t)
        if (t >= self.t_rest):
            self.mem += dt * (-self.mem + self.I * self.Rm) / self.tau_m
            if (self.mem > self.vth):
                self.spike = True
                self.spiketime.append(t)
                self.mem = 0
                self.t_rest = t + self.tau_ref

    def update_normal(self, dt, t):

        self.spike = False
        # self.I = self.I_ext
        if (t >= self.t_rest):
            self.mem += dt * (-self.mem + self.I * self.Rm) / self.tau_m
            if (self.mem > self.vth):
                self.spike = True
                self.mem = 0
                self.t_rest = t + self.tau_ref
                self.spiketime.append(t)

    def updateSynapses(self, t):
        for syn in self.synapses:
            syn.computeWeight(t)

    def setTestStates(self):
        self.t_rest = 0
        self.spiketime = []
        self.mem = 0
        self.I = 0
        self.I_ext = 0
        for syn in self.synapses:
            syn.strength = 0

        # print('I=' + str(self.I))

    def computeFilterCurrent(self):
        pass

    def setPreference(self):  # set preference of a neuron or called selectivity
        # self.selectivity = 2 * math.pi/16 * self.index - math.pi/16 # the mean of the Gaussian funtion
        pass

    def writeBasicInfoToJson(self, areaName):
        dic = {}
        dic["TrackID"] = self.layerIndex
        dic["GroupID"] = self.groupIndex
        dic["Index"] = self.index
        dic["area"] = areaName
        slist = []
        for syn in self.synapses:
            if (syn.weight <= 0): continue
            tmp = {}
            tmp["StrackID"] = syn.pre.layerIndex
            tmp["SgroupID"] = syn.pre.groupIndex
            tmp["Sindex"] = syn.pre.index
            tmp["Sarea"] = syn.pre.areaName
            tmp["TtrackID"] = self.layerIndex
            tmp["TgroupID"] = self.groupIndex
            tmp["Tindex"] = self.index
            tmp["Tarea"] = self.areaName
            tmp["type"] = syn.type
            tmp["weight"] = syn.weight
            slist.append(tmp)
        dic["synapses"] = slist
        return dic

    def writeSpikeTimeToJson(self):
        slist = []
        for i, t in enumerate(self.spiketime):
            dic = {}
            dic[i + 1] = t
            slist.append(dic)
        return slist

# neu  = LIFNeuron()
# dt = 0.001
# T = 1
# time = np.arange(0,T,dt)
# spikes = np.zeros(len(time))
# for i in range(0,len(time)):
#     if(i == 22):
#         print("debug")
#     neu.I = 84.49
#     neu.update_normal(dt, time[i])
#     if(neu.spike == True):
#         spikes[i] = 1
#         #spikes[i] = neu.mem
# print(len(neu.spiketime))
# pl.plot(time,spikes)
# pl.show()