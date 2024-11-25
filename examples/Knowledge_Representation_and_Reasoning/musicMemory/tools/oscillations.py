'''
Created on 2016.5.13

@author: liangqian
'''

from modal.lifneuron import LIFNeuron
from modal.cluster import Cluster
from modal.synapse import Synapse


c = Cluster('LIF')
c.createClusterNetwork()
c.setInhibitoryNeurons(0.2)

for i in range(0,c.neunum):
    for j in range(0,c.neunum):
        if(i != j):
            node = c.neurons[j]
            node.pre_neurons.append(c.neurons[i])
            syn = Synapse(c.neurons[i],node)


