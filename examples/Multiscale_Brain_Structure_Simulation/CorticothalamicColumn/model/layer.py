'''
Created on 2014.11.13.

@author: Liang Qian
'''

import sys
sys.setrecursionlimit(1000000)
from data import globaldata as Global
class Layer():
    '''
    layer class is used to build brain cortical layer of cortex
    class members are as follows:
    @param name: layer name(L1,L2,L3,.etc)
    @param neuronnum: total neural numbers of this layer
    @param neuraltype: neural type list of this layer
    @param neuronlist: different types of neuron list in this layer    
    '''


    def __init__(self):
        self.name = ''
        self.neuronnum = 0
        self.synapsenum = 0
        self.neuronlist = []
        self.synapselist = []
        self.neuronname = []
        
    def getLayerNeuronNumber(self):
        return len(self.neuronlist)
    def getLayerSynapseNumber(self):
        return len(self.synapselist)
    def getLayerNeuronTypes(self):
        return self.neuronname
        
    def stimulateNeuronInLayer4_BFS(self, T, neulist):
        for node in self.neuronlist:
            if(node.name == 'ss4(L2/3)'):
                break;
        step = int(T*1000/1.0)
        dc = 0
        for i in range(step):
            #print(i)
            strs = str(i)+','
            if(i > 1):
                for n in neulist:
                    if(n.totalindex == node.totalindex):continue
                    if(n.dc > 0):
                        n.integral(n.dc)
                        n.calc_spike()
                        if(n.spike == 1):
                            strs +=n.name+':'+str(n.totalindex)+','
            if(i < 10 or i > 25000):
                dc = 0
            else:
                dc = 400
            node.integral(dc)
            node.calc_spike()
            if(node.spike == 1):
                strs +=node.name+':'+str(node.totalindex)+','
            
            Global.f.write(strs+'\n')
        Global.f.close()