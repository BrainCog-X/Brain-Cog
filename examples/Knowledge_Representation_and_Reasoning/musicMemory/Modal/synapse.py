import math

class Synapse():
    '''
    classdocs
    '''

    def __init__(self, pre, post):
        '''
        Constructor
        '''
        self.type = 0  # 0: within a group; 1: different groups in the same layer; 2: other layer
        self.pre = pre
        self.post = post
        self.weight = 0
        self.excitability = 1  # 1:excited connection; 0:inhibitory connection
        self.strength = 0  # short term depression and facilitation factor
        self.delay = 0  # time delay of transmission between pre and post

    def computeWeight(self, t):

        if (self.type == 0):  # pre and post neurons are in the same group
            pass
        elif (self.type == 1):  # pre and post neurons are in the same layer but different groups
            for st in self.pre.spiketime:
                s = t - st
                temp = 0
                if (self.post.groupIndex - self.pre.groupIndex == self.delay):  # compute weight according to time delay
                    # using STDP rules
                    if (s >= 0):
                        temp = math.exp(-s / 5)
                    else:
                        #                         print(self.pre.groupIndex)
                        #                         print(self.post.groupIndex)
                        temp = -math.exp(s / 5)
                    self.weight += temp


        elif (self.type == 2):  # pre and post neurons are in the different layers
            pass
            # computing the STDP to update the weight within the time window

    def computeShortTermFacilitation(self, t):
        if (self.type == 1):
            for st in self.pre.spiketime[::-1]:
                at = st + self.delay
                if (at <= t and at >= t - self.post.tau_ref):  # between current time and time minus refractory period
                    temp = (self.strength + 1) * 0.2
                    self.strength += temp

        elif (self.type == 2):
            #             print(self.pre.areaName)
            #             print(self.pre.index)
            #             print(self.pre.groupIndex)
            for st in self.pre.spiketime:
                if (st <= t and st >= t - self.post.tau_ref):
                    temp = (self.strength + 1) * 0.5
                    self.strength = self.strength + temp

        elif (self.type == 0):
            if (self.excitability == 0):
                for st in self.pre.spiketime:
                    self.strength += (self.strength + 1) * 0.8

    def computeShortTermReduction(self, t):
        if (self.type == 2):
            for st in self.pre.spiketime[::-1]:
                if (t - st > self.post.timeWindow):
                    self.strength -= (self.strength + 1) * 0.5