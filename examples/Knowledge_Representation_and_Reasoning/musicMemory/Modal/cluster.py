
from .lifneuron import LIFNeuron
from .synapse import Synapse
from Modal.izhikevichneuron import *

class Cluster():
    '''
    classdocs
    '''

    def __init__(self, neutype='LIF', neunum=10):
        '''
        Constructor
        '''
        self.id = 0  # starting with 1
        self.name = ''  # name of this group
        self.neutype = neutype
        self.neunum = neunum
        self.neurons = []
        self.timeWindow = 5  # ms

    def createClusterNetwork(self):
        # create neurons
        for i in range(0, self.neunum):
            if (self.neutype == 'LIF'):
                node = LIFNeuron()
                node.index = i + 1
                node.setPreference()
                self.neurons.append(node)
            if (self.neutype == 'Izhikevich'):
                node = IzhikevichNeuron()
                node.index = i
                self.neurons.append(node)
            if (self.neutype == 'Gaussian'):
                node = GaussianNeuron()
                node.index = i + 1
                self.neurons.append(node)
            if (self.neutype == 'HH'):
                node = HHNeuron()
                node.index = i
                self.neurons.append(node)

    def setInhibitoryNeurons(self, ratio_inhneuron):
        for i in range(int(self.neunum * (1 - ratio_inhneuron)), self.neunum):
            self.neurons[i].type = 'inh'

    def setPropertiesofNeurons(self, groupID, layerType, layerID):
        for n in self.neurons:
            n.layerType = layerType
            n.groupIndex = groupID
            n.layerIndex = layerID

    def setTestStates(self):
        for neu in self.neurons:
            neu.setTestStates()

    def createFullConnections(self):  # all in all connections
        for i in range(0, self.neunum):
            neu = self.neurons[i]
            for j in range(0, self.neunum):
                if (j != i):
                    syn = Synapse(self.neurons[j], neu)  # this neuron is considered as post_synapse neuron
                    syn.type = 0
                    neu.synapses.append(syn)
                    neu.pre_neurons.append(self.neurons[j])

    def createInhibitoryConnections(self):  # all in all inhibitory connections
        for i in range(0, self.neunum):
            neu = self.neurons[i]
            for j in range(0, self.neunum):
                if (j != i):
                    syn = Synapse(self.neurons[j], neu)  # this neuron is considered as post_synapse neuron
                    syn.type = 0
                    syn.excitability = 0
                    syn.weight = 20
                    neu.synapses.append(syn)
                    neu.pre_neurons.append(self.neurons[j])

    def writeSelfInfoToJson(self):
        dic = {}
        nlist = []
        for neu in self.neurons:
            if (len(neu.spiketime) <= 0): continue
            ndic = neu.writeBasicInfoToJson()
            nlist.append(ndic)
        dic["GroupID"] = self.id
        dic["Name"] = self.name
        dic["Neuron"] = nlist
        return dic

    def writeSpikeInfoToJson(self):
        nlist = []
        for neu in self.neurons:
            if (len(neu.spiketime) > 0):
                tmp = {}
                tmp["GroupID"] = neu.groupIndex
                tmp["Index"] = neu.index
                tmp["SpikeTime"] = neu.writeSpikeTimeToJson()
                nlist.append(tmp)
        return nlist

class ModeCluster(Cluster):
    def __init__(self, neutype, neunum):
        Cluster.__init__(self, neutype,neunum)

    def createClusterNetwork(self, areaName):
        for i in range(self.neunum): # 暂时先不考虑importance
            if (self.neutype == 'LIF'):
                node = ModeLIFNeuron()
                node.index = i + 1
                node.areaName = areaName
                node.selectivity = i+1
                self.neurons.append(node)
            if (self.neutype == 'Izhikevich'):
                self.neutype = 'Izhikevich'
                node = ModeIzhikevichNeuron()
                node.index = i + 1
                node.areaName = areaName
                node.selectivity = i+1
                self.neurons.append(node)

class KeyCluster(Cluster):
    def __init__(self, neutype, neunum):
        '''
        Constructor
        '''
        Cluster.__init__(self, neutype, neunum)

    def createClusterNetwork(self,tone,areaName):
        for i in range(0, self.neunum):
            if (self.neutype == 'LIF'):
                node = KeyLIFNeuron()
                node.index = i + 1
                node.areaName = areaName
                node.selectivity = i
                node.importance = tone[i]
                self.neurons.append(node)
            if(self.neutype == 'Izhikevich'):
                self.neutype = 'Izhikevich'
                a=0
                b=0
                c=0
                d=0
                if tone[i] == 2:
                    a = 0.02
                    b = 0.2
                    c = -55
                    d = 4
                if tone[i] == -1:
                    a=0.1
                    b = 0.2
                    c = -65
                    d = 2
                if(tone[i] == 1):
                    a = 0.02
                    b = 0.2
                    c = -65
                    d = 8
                node = KeyIzhikevichNeuron(a,b,c,d)
                node.index = i + 1
                node.areaName = areaName
                node.selectivity = i
                node.importance = tone[i]
                self.neurons.append(node)

class ChordCluster(Cluster):
    def __init__(self, neutype,neunum):
        Cluster.__init__(self, neutype, neunum)

    def createClusterNetwork(self):
        for i in range(self.neunum):
            if (self.neutype == 'LIF'):
                node = ChordLIFNeuron()
                node.index = i + 1
                node.areaName = 'Chord'
                node.selectivity = i
                node.importance = 1
                self.neurons.append(node)
            if self.neutype == 'Izhikevich':
                node = ChordIzhikevichNeuron()
                node.index = i + 1
                node.areaName = 'Chord'
                node.selectivity = i
                node.importance = 1
                self.neurons.append(node)