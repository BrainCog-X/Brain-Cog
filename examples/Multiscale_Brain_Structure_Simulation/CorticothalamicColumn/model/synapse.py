'''
Created on 2014.11.13

@author: Liang Qian
'''

class Synapse():
    '''
    synapsis class is used to create a synapsis structure
    members are as follows:
    @param pre: pre-synapsis neuron
    @param post: post-synapsis neuron
    @param locationlayer: layer where this synapse locate in   
    '''


    def __init__(self, pre,post,locationlayer):
        self.pre = pre
        self.post = post
        self.locationlayer = locationlayer
        self.I = 0
        self.weight = 0 if(pre.name == 'p2/3' and post.name == 'p2/3') else -1
#         self.tauAMPA = 5
#         self.tauNMDA = 150
#         self.tauGABAA = 6
#         self.tauGABAB = 150
#         self.STDPA_pos = 1
#         self.STDPA_neg = 2
#         self.tau_pos = 20
#         self.tau_neg = 20
    def getInfo(self,f,nodename,denpos,denlayer):
        f.write('neuron:'+nodename+','+'dendrite:'+denpos+','+'den_layer:'+denlayer+','
            +'syn_Layer:'+ self.locationlayer+','
            + 'pre_neuron:'+self.pre.name+','+'pre_neuron_index:'+str(self.pre.totalindex)+','
            + 'post_neuron:'+self.post.name+','+'post_neuron_index:'+str(self.post.totalindex)+','
            + 'weight:'+str(self.weight)+'\n')    
        
        