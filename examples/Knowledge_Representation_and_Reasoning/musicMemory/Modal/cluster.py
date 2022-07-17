
from .lifneuron import LIFNeuron
from .synapse import Synapse


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
            # if (self.neutype == 'Izhi'):
            #     node = IzhikevichNeuron(a=0.02, b=0.2, c=-65, d=8, vthresh=30)
            #     node.index = i
            #     self.neurons.append(node)
            # if (self.neutype == 'Gaussian'):
            #     node = GaussianNeuron()
            #     node.index = i + 1
            #     self.neurons.append(node)
            # if (self.neutype == 'HH'):
            #     node = HHNeuron()
            #     node.index = i
            #     self.neurons.append(node)

    def setInhibitoryNeurons(self, ratio_inhneuron):
        for i in range(int(self.neunum * (1 - ratio_inhneuron)), self.neunum):
            self.neurons[i].type = 'inh'

    def setPropertiesofNeurons(self, groupID, layerType, layerID):
        for n in self.neurons:
            n.layerType = layerType
            n.groupIndex = groupID
            n.layerIndex = layerID

    def setTestStates(self):
        if (self.neutype == 'LIF'):
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

    def writeSelfInfoToJson(self, areaName):
        dic = {}
        nlist = []
        for neu in self.neurons:
            if (len(neu.spiketime) <= 0): continue
            ndic = neu.writeBasicInfoToJson(areaName)
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
