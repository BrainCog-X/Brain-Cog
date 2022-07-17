'''
Created on 2014.11.25

@author: Liang Qian
'''

#from tools import dbconnection as DB
from tools import exdata as Data

data = Data.EXDATA()
curneuronindex = 0


SynapseNumberPerDendrite = 40
ProximalDendriteNumerPerNeuron = 1
f = open('fire.csv','w')

