from .layer import Layer
from .composercluster import ComposerCluster


class ComposerLayer(Layer):
    '''
        This layer defines the information of composer name. One neuron corresponds to a composer
    '''

    def __init__(self, neutype='LIF'):
        self.neutype = neutype
        self.groups = {}

    def setTestStates(self):
        for id, g in self.groups.items():
            g.setTestStates()

    def addNewGroups(self, groupID, layerID, neunum, composername):
        g = ComposerCluster(self.neutype, neunum)
        g.id = groupID
        g.name = composername
        g.createClusterNetwork()
        g.setPropertiesofNeurons(groupID, 'G', layerID)
        self.groups[composername] = g