from .cluster import Cluster
from .tempolifneuron import TempoLIFNeuron


class TempoCluster(Cluster):
    '''
    classdocs
    '''

    def __init__(self, neutype, neunum):
        '''
        Constructor
        '''
        Cluster.__init__(self, neutype, neunum)

    def createClusterNetwork(self):
        for i in range(0, self.neunum):
            if (self.neutype == 'LIF'):
                node = TempoLIFNeuron()
                node.index = i + 1
                node.areaName = 'TMSM'
                node.setPreference()
                self.neurons.append(node)