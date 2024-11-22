from .layer import Layer
from .cluster import Cluster
from .synapse import Synapse


class SequenceLayer(Layer):
    '''
    This class mainly stores the musical sequential elements, including pitches and durations
    '''

    def __init__(self, neutype='LIF'):
        '''
        Constructor
        '''
        self.type = ""
        self.neutype = neutype
        self.groups = {}

    def addNewGroups(self, GroupID, layerID, neunum):
        g = Cluster(self.neutype, neunum)
        g.createClusterNetwork()
        g.id = GroupID
        g.setPropertiesofNeurons(g.id, 'S', layerID)
        self.groups[g.id] = g

        # create full connection with the former group
        if (len(self.groups) > 1):
            for i in range(1, g.id)[::-1]:
                pre_g = self.groups.get(i)
                for n1 in pre_g.neurons:
                    for n2 in g.neurons:
                        if (n1.type == 'inh' or n2.type == 'inh'): continue;
                        syn = Synapse(n1, n2)
                        syn.type = 1
                        syn.delay = g.id - pre_g.id
                        n2.pre_neurons.append(n1)
                        n2.synapses.append(syn)

    def setTestStates(self):
        for gid, g in self.groups.items():
            g.setTestStates()