

from .layer import Layer
from .titlecluster import TitleCluster


class TitleLayer(Layer):
    '''
    classdocs
    '''

    def __init__(self, neutype='LIF'):
        '''
        Constructor
        '''
        self.neutype = neutype
        self.groups = {}

    def setTestStates(self):
        for id, g in self.groups.items():
            g.setTestStates()

    def addNewGroups(self, groupID, layerID, neunum, goalname):
        g = TitleCluster(self.neutype, neunum)
        g.id = groupID
        g.name = goalname
        g.createClusterNetwork()
        g.setPropertiesofNeurons(groupID, 'G', layerID)
        self.groups[goalname] = g