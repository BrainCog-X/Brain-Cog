'''
Created on 2015.5.27

@author: Liang Qian
'''

from data import globaldata as Global
from .layer import Layer
from .synapse import Synapse

class Cortex():
    '''
    This class defines properties and funtions of cortex
    '''


    def __init__(self,neuronnumscale):
        '''
        Constructor
        '''
        self.neuronnumscale = neuronnumscale
        self.neuronsNumber = 0
        self.synapsesNumber = 0
        self.neurons = [] # a list storing all neurons of the whole cortex
        self.layers = {} # a dictionary storing layers of cortex
        self.synapses = [] # a list storing all synapses of the whole cortex
        self.minicolumns = [] # a list storing information per mini-column
        self.neurontoindex = {} # a dictionary storing name to index of neuronlist
        self.totaldata = Global.data.getCortexData()
    
    def setNeuronToIndex(self,node):
        name = node.name
        if(self.neurontoindex.get(name) == None):
            self.neurontoindex[name] = len(self.neurons) - 1
    def setLayers(self):
        layerdic = Global.data.getLayerData()
        for i,info in layerdic.items():
            layer = Layer()
            layer.name = info.get('name')
            layer.neuronnum = self.neuronnumscale * float(info.get('neuronnum'))/100
            print(layer.name + ' neuron number:'+str(layer.neuronnum))
#             layer.synapsenum = self.synapsenum * info.get('synapsenum')
#             print(layer.name + ' synapse number:'+str(layer.synapsenum))
            self.layers[layer.name] = layer

