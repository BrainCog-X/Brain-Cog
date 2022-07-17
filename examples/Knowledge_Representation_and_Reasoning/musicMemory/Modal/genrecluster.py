from .cluster import Cluster
from .genrelifneuron import GenreLIFNeuron

class GenreCluster(Cluster):
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
                node = GenreLIFNeuron()
                node.index = i + 1
                node.areaName = 'Genre'
                self.neurons.append(node)