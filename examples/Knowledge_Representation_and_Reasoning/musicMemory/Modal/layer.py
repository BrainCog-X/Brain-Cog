from abc import ABCMeta,abstractmethod
from conf.conf import configs
from Modal.cluster import *
class Layer():
    '''
    classdocs
    '''
    _metaclass_ = ABCMeta

    def __init__(self, neutype):
        '''
        Constructor
        '''
        self.neutype = neutype
        self.groups = {}

    @abstractmethod
    def resetProperties(self):
        raise NotImplementedError

    def addNewGroups(self, layerID, neunum):
        raise NotImplementedError

class ModeLayer(Layer):
    def __init__(self, neutype = 'LIF'):
        self.neutype = neutype
        self.groups = {}

    def setTestStates(self):
        for id, g in self.groups.items():
            g.setTestStates()

    def addNewGroups(self, groupID, layerID, neunum, modeName):
        g = ModeCluster('Izhikevich', neunum)
        g.id = groupID
        g.name = modeName
        g.createClusterNetwork(g.name)
        g.setPropertiesofNeurons(groupID,'Mode',layerID)
        self.groups[groupID-1] = g

class KeyLayer(Layer):
    def __init__(self, neutype='LIF'):
        self.neutype = neutype
        self.groups = {}

    def setTestStates(self):
        for id, g in self.groups.items():
            g.setTestStates()

    def addNewGroups(self, groupID, layerID, neunum, key):
        g = KeyCluster('Izhikevich', neunum)
        g.id = groupID
        g.name = configs.index2key.get(groupID-1)
        g.createClusterNetwork(key,g.name)
        g.setPropertiesofNeurons(groupID, 'Key', layerID)
        self.groups[groupID-1] = g

class ChordLayer(Layer):
    def __init__(self, neutype = 'LIF'):
        Layer.__init__(self,neutype)

    def setTestStates(self):
        for id, g in self.groups.items():
            g.setTestStates()

    def addNewGroups(self,groupID, layerID, neunum):
        g = ChordCluster('Izhikevich', neunum)
        g.id = groupID
        g.name = groupID
        g.createClusterNetwork()
        g.setPropertiesofNeurons(groupID, 'Chord', layerID)
        self.groups[groupID - 1] = g