from .cluster import Cluster
from .composerlifneuron import ComposerLIFNeuron


class ComposerCluster(Cluster):
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
                node = ComposerLIFNeuron()
                node.index = i + 1
                node.areaName = 'Composer'
                self.neurons.append(node)