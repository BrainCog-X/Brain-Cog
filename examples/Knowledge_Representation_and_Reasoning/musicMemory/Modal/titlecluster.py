from .cluster import Cluster
from .titlelifneuron import TitleLIFNeuron


class TitleCluster(Cluster):
    '''
    classdocs
    '''

    def __init__(self, neutype, neunum):
        '''
        Constructor
        '''
        Cluster.__init__(self, neutype, neunum)
        self.averageFiringRate = 0

    def createClusterNetwork(self):
        for i in range(0, self.neunum):
            if (self.neutype == 'LIF'):
                node = TitleLIFNeuron()
                node.index = i + 1
                node.areaName = 'IPS'
                self.neurons.append(node)
#             if(self.neutype == 'Izhi'):
#                 node = IzhikevichNeuron(a = 0.02,b = 0.2,c = -65,d = 8,vthresh = 30)
#                 node.index = i
#                 self.neurons.append(node)
#             if(self.neutype == 'Gaussian'):
#                 node = GaussianNeuron()
#                 node.index = i+1
#                 self.neurons.append(node)
#             if(self.neutype == 'HH'):
#                 node = HHNeuron()
#                 node.index = i
#                 self.neurons.append(node)