'''
Created on 2014.11.13

@author: Liang Qian
'''

import sys
from .dendrite import Dendrite
sys.path.append('../')

from data import globaldata as Global
from .thalamus import Thalamus
from .cortex import Cortex
from .layer import Layer
from .synapse import Synapse
from braincog.base.node.node import *
class Cortex_Thalamus():
    '''
    cortex class is used to build human brain cortex
    members are as follows:
    @param neuronnum: total neuron number
    @param layer: layer list
    
    at the aspect of coding or algorithm,cortex is a huge Graph,neurons can be seen as nodes,synapses can be seen as edges,
    but there may be plenty of edges between any two given nodes,so we can not use the <ni,nj> to express an edge,
    the edge should be defined as an object,if we use adjacency list to store this huge graph,the form is just like
    node->map(node,list<edge>)->(map(another node,list<edge>))
    '''


    def __init__(self, neuronnumscale):
        self.neuronnumscale = neuronnumscale
        self.cortex = Cortex(neuronnumscale)
        self.thalamus = Thalamus()
        self.minicolumns = [] # a list storing information per mini-column
        self.synapsenum = 0
        self.neuronname = [] # a list storing neuron name in all cortex
        self.neuron = []  # a list storing all neuron in cortex
        self.synapse = [] # a list storing all synapses in thalamocortex
        self.neurontoindex = {} # a dictionary to storing the neuron mapping to index in neuron list
        self.totaldata = Global.data.getCortexData()
        
        
    def setSynapseNum(self):
        num = 0
        for name,info in self.totaldata.items():
            num += self.neuronnumscale * (info.get('synnum')*info.get('neunum')/100.0)
            self.neuronname.append(info.get('neuronname'))
        self.synapsenum = num
        
    def setLayer(self):
        layerdic = Global.data.getLayerData()
        for i,info in layerdic.items():
            layer = Layer()
            layer.name = info.get('name')
            layer.neuronnum = self.neuronnumscale * info.get('neuronnum')/100
            print(layer.name + ' neuron number:'+str(layer.neuronnum))
            layer.synapsenum = self.synapsenum * info.get('synapsenum')
            print(layer.name + ' synapse number:'+str(layer.synapsenum))
            self.layer[layer.name] = layer        
 
    def setNeuronsDendritesAndSynapes(self): 
        neurondic = Global.data.getNeuronData()
        index = 0
        for s in self.neuronname:
            neuroninfo = neurondic.get(s)
            self.neurontoindex[s] = len(self.neuron)
            num = self.neuronnumscale * float(neuroninfo.get('percent'))/100.0
            #using synapse numbers to compute dendrites numbers
            synapsedic = Global.data.getSynapseData(s)
            dendic = {}
            totaldennum = 0
            for r,item in synapsedic.items():
                synum = int(item.get('synapsenum'))
                dennum = (synum+Global.SynapseNumberPerDendrite-1)//Global.SynapseNumberPerDendrite # a dendrite contains no more than 40 synapses
                loc = item.get('locationlayer')
                dendic[loc] = dennum
            for i in range(int(num)):
                #init
                node = CTIzhNode(morphology = neuroninfo.get('morphology'),
                                name = neuroninfo.get('name'),
                                excitability = neuroninfo.get('excitability'),
                                spiketype = neuroninfo.get('spiketype'),
                                synnum = synum,
                                locationlayer = neuroninfo.get('location layer'),
                                totalindex = index,
                                Gup = float(neuroninfo.get('Gup')),
                                Gdown = float(neuroninfo.get('Gdown')),
                                Vr = float(neuroninfo.get('Vr')),
                                Vt = float(neuroninfo.get('Vt')),
                                Vpeak = float(neuroninfo.get('Vpeak')),
                                a = float(neuroninfo.get('a')),
                                b = float(neuroninfo.get('b')),
                                c = float(neuroninfo.get('Csoma')),
                                d = float(neuroninfo.get('d')),
                                capacitance = float(neuroninfo.get('capacitance')),
                                k = float(neuroninfo.get('k')),
                                 )
                # set dendrites
                count = 0; flag = False
                for dlocatelayer,dnum in dendic.items():
                    for j in range(dnum):
                        den = Dendrite()
                        den.locationlayer = dlocatelayer
                        if(dlocatelayer != node.locationlayer or flag):
                            den.postion = 'distal'
                            node.distal_dendrites.append(den)
                        else:
                            if(count < Global.ProximalDendriteNumerPerNeuron):
                                den.postion = 'proximal'
                                node.proximal_dendrites.append(den)
                                count += 1
                                if(count >= Global.ProximalDendriteNumerPerNeuron):
                                    flag = True                                                    
                #node.getDendritesInfo()
                if(node.locationlayer == 'T'):
                    self.thalamus.neuronsNumber += 1
                    self.thalamus.neurons.append(node)
                    self.thalamus.setNeuronToIndex(node)
                else: 
                    self.cortex.neuronsNumber += 1
                    self.cortex.neurons.append(node)
                    self.cortex.setNeuronToIndex(node)
                    la = self.cortex.layers.get(node.locationlayer)
                    la.neuronlist.append(node)
                    if node.name not in la.neuronname:
                        la.neuronname.append(node.name)
                self.neuron.append(node)
                index += 1
        # set synapses
        for postindex,node in enumerate(self.neuron): # this step can be optimized
            synapsedic = Global.data.getSynapseData(node.name)
            # get synapse_pre neuron and relative numbers of synapses
            count = 0
            for r,item in synapsedic.items():
                #print(item)
                for s in self.neuronname:
                    #print("pre_neuron_name:"+s)
                    totalsynapsenum = round(item.get('synapsenum') * item.get(s)/100.0)
                    if(postindex == 0):count += totalsynapsenum
                    #print("pre_neuron_name_synapse_num:"+str(totalsynapsenum))
                    if(totalsynapsenum > 0): #if this neuron connect to synapse_pre neuron s,distribute the synapse to these neurons
                        info = self.totaldata.get(s)
                        preneuronnum = round(self.neuronnumscale * int(info.get('neunum'))/100)
                        #print("pre_neuron_name_number:" + str(preneuronnum))
                        avgnum = lastnum = 0
                        if(preneuronnum == 1):
                            avgnum = lastnum = totalsynapsenum
                        else:
                            avgnum = totalsynapsenum // (preneuronnum-1)
                            lastnum = totalsynapsenum - (avgnum*(preneuronnum-1))
                        preneuronindex = self.neurontoindex.get(s)
                        for j in range(preneuronindex,preneuronindex+preneuronnum):
                            if(j == postindex):lastnum += avgnum;continue # not connect to itself
                            preneuron = self.neuron[j]
                            synapselist = []
                            if(preneuron.adjneuronlist.get(node) != None):
                                synapselist = preneuron.adjneuronlist.get(node)
                            if(j == preneuronindex+preneuronnum-1): #the last neuron
                                for t in range(lastnum):
                                    synapse = Synapse(self.neuron[j],node,item.get('locationlayer'))
                                    synapselist.append(synapse)
                                    self.synapse.append(synapse)
                                    if(item.get('locationlayer') != 'T'):
                                        layerinfo = self.cortex.layers.get(item.get('locationlayer'))
                                        layerinfo.synapselist.append(synapse)
                                    if(node.locationlayer == 'T'): 
                                        self.thalamus.synapses.append(synapse)
                                        self.thalamus.synapsesNumber += 1
                                    else:
                                        self.cortex.synapses.append(synapse)
                                        self.cortex.synapsesNumber += 1
                            else:
                                for t in range(avgnum):
                                    synapse = Synapse(self.neuron[j],node,item.get('locationlayer'))
                                    synapselist.append(synapse)
                                    self.synapse.append(synapse)
                                    if(item.get('locationlayer') != 'T'):
                                        layerinfo = self.cortex.layers.get(item.get('locationlayer'))
                                        layerinfo.synapselist.append(synapse)
                                    if(node.locationlayer == 'T'): 
                                        self.thalamus.synapses.append(synapse)
                                        self.thalamus.synapsesNumber += 1
                                    else:
                                        self.cortex.synapses.append(synapse)
                                        self.cortex.synapsesNumber += 1
                            if(preneuron.adjneuronlist.get(node) == None and len(synapselist) > 0):
                                preneuron.adjneuronlist[node] = synapselist            
                      
        # set these synapses to dendrite list
        #self.setSynapsesToDendrites()
    def setSynapsesToDendrites(self):
        for node in self.neuron:
            if node.name == 'TCs': # synapses from TCs to ss4(L4) must be located in proximal dendrites of ss4(L4)
                for post,synlist in node.adjneuronlist.items():
                    if(post.name == 'ss4(L4)'):
                        for syn in synlist:
                            flag = post.addSynapseToDendrite('proximal',syn)
                            if(not flag):flag = post.addSynapseToDendrite('distal',syn)
                            if(not flag):
                                print('all dendrites are full in neuron' + post.name + '_'+str(node.totalindex))
                    else:
                        for syn in synlist:
                            flag = False
                            if(post.locationlayer == syn.locationlayer):
                                flag = post.addSynapseToDendrite('proximal',syn)
                                if(not flag):
                                    flag = post.addSynapseToDendrite('distal',syn)
                            else:
                                flag = post.addSynapseToDendrite('distal',syn)
                            if(not flag): print('all dendrites are full in neuron' + post.name + '_'+str(node.totalindex))
            else:
                for post,synlist in node.adjneuronlist.items():
                    for syn in synlist:
                        flag = False
                        if(post.locationlayer == syn.locationlayer):
                            flag = post.addSynapseToDendrite('proximal',syn)
                            if(not flag):
                                flag = post.addSynapseToDendrite('distal',syn)
                            if(not flag): print('all dendrites are full in neuron' + node.name + '_'+str(node.totalindex))
#         f = open('dendrites_info.csv','w')
#         for node in self.neuron:
#             if(node.totalindex == 0):
#                 node.getDendriteSynapsesInfo(f)
#         f.close()

    def setCortexProperties(self):
        self.cortex.setCortexProperties()
    def setThalamusProperties(self):
        self.thalamus.setThalamusProperties()            
    
    def CreateCortexNetwork(self):
        self.setSynapseNum()
        self.cortex.setLayers()
        self.setNeuronsDendritesAndSynapes()
        self.setThalamusProperties()

                
    #----------API Of the whole network--------------#
    def getTotalNeuronNumber(self):
        return len(self.neuron)
    def getTotalSynapseNumber(self):
        return len(self.synapse)
    def getCortexNeuronNumber(self):
        return self.corticalneuronnumber
    def getThalamoNeuronNumber(self):
        return self.thamlamoneuronnumber
    def getSpecifiedNeuronNumber(self,name):
        result = {}
        if name in self.neuronname:
            info = self.totaldata.get(name)
            num = self.neuronnumscale * info.get('neunum')/100
            result[name] = num
        elif name == 'all':
            for r,info in self.totaldata.items():
                num = self.neuronnumscale * info.get('neunum')
                result[r] = num
        return result   
    def getNeuronTypesNumber(self):
        return len(self.neuronname)
    def getNeuronTypes(self):
        return self.neuronname
    def getCorticalSynapseNumber(self):
        return len(self.corticalsynapse)
    def getThalamoSynapseNumber(self):
        return len(self.corticalsynapse)
    def getPreAndPostNeuronsOfSynapse(self,index):
        if(index >= 0 or index <= len(self.synapse -1)):
            return self.synapse[index].pre,self.synapse[index].post
        else: return None
    #--------------API of the layer----------------------#
    def getCortexLayerNeuronNumber(self,layername):
        layerinfo = self.layer.get(layername)
        return layerinfo.getLayerNeuronNumber()
    def getCortexLayerSynapseNumber(self,layername):
        layerinfo = self.layer.get(layername)
        return layerinfo.getLayerSynapseNumber()
    def getCortexLayerNeuronTypes(self,layername):
        layerinfo = self.layer.get(layername)
        if(layerinfo == None):
            print(layername +" is not in Cortex!")
            return None
        return layerinfo.neuronname                                
    def getCortexLayerPreAndPostNeuronsOfSynapse(self,layername,index):
        layerinfo = self.layer.get(layername)
        if(index >= 0 and index < len(layerinfo.synapselist)):
            return layerinfo.synapselist[index].pre,  layerinfo.synapselist[index].post
    def getNeuronAllPreNeuronsTypes(self,index):
        if(index >= 0 and index < len(self.neuron)):
            node = self.neuron[index]
            return node.getWholePreSynapseNeuronType()
    def outputNeuronInfo(self):
        f = open('neuron.csv','w')
        f.write('index,name,morphology,excitability,locationlayer\n')
        for node in self.neuron:
            flag = 'No'
            if(node.excitability == "TRUE"):
                flag = 'Yes'           
            f.write(str(node.totalindex)+','+node.name+','+node.morphology+','+flag+','+node.locationlayer+'\n')
        f.close()
    def outputConnectionMatrix(self):
        M = len(self.neuron)
        matrix = [[0 for col in range(M)] for row in range(M)]
        for node in self.neuron:
            for post,list in node.adjneuronlist.items():
                weight = len(list)
                row = node.totalindex
                col = post.totalindex
                matrix[row][col] = weight
        f = open('connection.csv','w')
        name = ''
        for node in self.neuron:
            name += node.name + ','
        f.write(name+'\n')
        for row in range(M):
            line = ''
            for col in range(M):
                line += str(matrix[row][col])+','
            f.write(line+'\n')
        f.close()
    def outputsynapspercent(self,namelist):
        totalcount = 0
        slist = {'L1':0,'L2/3':0,'L4':0,'L5':0,'L6':0,'T':0}
        for name in namelist:
            for node in self.neuron:
                for pre,list in node.adjneuronlist.items():
                    if pre.name == name:
                        totalcount = totalcount+len(list)
                        loclayer = node.locationlayer
                        value = slist.get(loclayer) + len(list)
                        slist[loclayer] = value
        print(slist)
#-----------------------runing the whole network---------------------------#
    def run(self):
        '''
        run the cortical system
        '''
        #s1:stimulate the neuron in L4
        L = self.cortex.layers.get('L4')
        L.stimulateNeuronInLayer4_BFS(30,self.neuron)



    def outputSpikeThreashold(self):
        f = open('threashold.csv','w')
        for node in self.neuron:
            f.write(str(node.Vpeak)+'\n')
        f.close() 


            
        
                     
                                            
                