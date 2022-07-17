'''
Created on 2015.5.27

@author: Liang Qian
'''

class Thalamus():
    '''
    This class defines the basic functions and properties of thalamus
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.neuronsNumber = 0
        self.synapsesNumber = 0
        self.neurons = []
        self.synapses = []
        self.neurontoindex = {}
    
    def setNeuronToIndex(self,node):
        name = node.name
        if(self.neurontoindex.get(name) == None):
            self.neurontoindex[name] = len(self.neurons)-1
    def setThalamusProperties(self):
        print(len(self.synapses))
        for node in self.neurons:
            if(node.name == 'TCs'):
                for post,synlist in node.adjneuronlist.items():
                    if(post.name == 'ss4(L2/3)'):
                        for syn in synlist:
                            syn.weight = 0
            if(node.name == 'TCn'):
                for post,synlist in node.adjneuronlist.items():
                    if(post.name == 'p6(L5/6)'):
                        for syn in synlist:
                            syn.weight = 0
    