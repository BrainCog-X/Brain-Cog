from .sequencelayer import SequenceLayer
from .notecluster import NoteCluster
from .synapse import Synapse


class NoteSequenceLayer(SequenceLayer):
    '''
    classdocs
    '''

    def __init__(self, neutype):
        '''
        Constructor
        '''
        SequenceLayer.__init__(self, neutype)

    def addNewGroups(self, GroupID, layerID, neunum):
        g = NoteCluster(self.neutype, neunum)
        g.createClusterNetwork()
        # g.createInhibitoryConnections()
        g.id = GroupID
        g.setPropertiesofNeurons(g.id, 'S', layerID)
        self.groups[g.id] = g

        # create full connection with the former group
        if (len(self.groups) > 1):
            s = 0
            if (g.id <= 5):
                s = 1
            else:
                s = g.id - 4
            # for i in range(1,g.id)[::-1]:
            for i in range(s, g.id)[::-1]:
                pre_g = self.groups.get(i)
                for n1 in pre_g.neurons:
                    for n2 in g.neurons:
                        if (n1.type == 'inh' or n2.type == 'inh'): continue;
                        syn = Synapse(n1, n2)
                        syn.type = 1
                        syn.delay = g.id - pre_g.id
                        n2.pre_neurons.append(n1)
                        n2.synapses.append(syn)