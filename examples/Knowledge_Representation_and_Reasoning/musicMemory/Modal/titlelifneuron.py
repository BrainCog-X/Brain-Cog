from .lifneuron import LIFNeuron
import math


class TitleLIFNeuron(LIFNeuron):
    '''
    classdocs
    '''

    def __init__(self, tau_ref=0, vthresh=5, Rm=2, Cm=0.2):
        '''
        Constructor
        '''
        LIFNeuron.__init__(self, tau_ref, vthresh, Rm, Cm)

    def updateCurrentOfLowerAndUpperLayer(self, t):
        self.I_lower = 0
        self.I_upper = 0
        for syn in self.synapses:
            syn.computeShortTermFacilitation(t)
            syn.computeShortTermReduction(t)
            if (syn.type == 2):  # pre and post neurons come from the different layers
                self.I_lower += syn.weight * syn.strength

        if (self.I_lower <= 0):
            self.I = self.I_lower
        else:
            self.I = math.log(self.I_lower)

    def update(self, dt, t):
        self.spike = False
        # self.updateCurrentOfLowerAndUpperLayer(t)
        if (t >= self.t_rest):
            self.mem += dt * (-self.mem + self.I * self.Rm) / self.tau_m
            if (self.mem > self.vth):
                self.spike = True
                self.spiketime.append(t)
                self.mem = 0
                self.t_rest = t + self.tau_ref

    def computeFiringRate(self):
        if (self.I == 0):
            self.firingrate = 0
        else:

            self.firingrate = 1 / (self.tau_ref + self.Rm * self.Cm * math.log(self.I / (self.I - self.vth)))
            self.firingrate *= 1000
            self.firingrate = round(self.firingrate)
            # print(self.firingrate)