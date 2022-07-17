'''
Created on 2015.5.19

@author: liangqian
'''
import sys

import sys
from data.globaldata import *

class Dendrite():
    '''
    This class defines dendrite structure of a neuron.
    A dendrite contains no more than 40 synapses
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.synapses = [] # synapses list which this dendrite contains
        self.locationlayer = '' # layer this dendrite locates in
        self.postion = '' # the distance from soma, proximal or distal
    
    def setSynapse(self,syn):
        '''
            This function is going to insert the Synapse syn to this dendrite
            if the number of synapse of this dendrite is more than theshold, the current synapse
            can not be inserted to the dendrite.
        '''
        if(len(self.synapses) >= SynapseNumberPerDendrite):
            return False
        else:
            self.synapses.append(syn)
            return True
    def getSynapseInfo(self,f,nodename,denpos):
        for syns in self.synapses:
            syns.getInfo(f,nodename,denpos,self.locationlayer)    