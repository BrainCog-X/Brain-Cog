import numpy as np

device='cuda:0'
import matplotlib.colors as mcolors
import torch
import random
import sys
from random import sample
from spikingjelly.clock_driven import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt
from random import choice,sample
sys.path.append("..")
from tools.MazeTurnEnvVec import *
import math
from tools.LSM_helper import calc_priority_based_on_dis,update_matrix_to_list,draw_spikes,compute_rank,population
from tools.update_weights import stdp,bcm,regul

class LSM(object):

    def __init__(self, n_offsprings=20,seed=0,
                 height=8, width=8,
                 input_size=4, output_size=3,
                 stp_alpha=0.01, stp_beta=0.3, w_input_scale=1,w_liquid_scale=4,w_output_scale=6,primary_amount=5,secondary_amount=5,
                 I_Vth=35,liquid_density=0.1,
                 delay_device=None):
        self.w_liquid_scale=w_liquid_scale
        self.n_offsprings=n_offsprings
        self.track_data=False
        self.n_input = input_size
        self.n_output=output_size
        self.width = width
        self.height = height
        self.out=None
        self.stp_alpha = stp_alpha
        self.stp_beta = stp_beta
        self.num_of_liquid_layer_neurons = width * height
        self.priority=calc_priority_based_on_dis(num=self.num_of_liquid_layer_neurons,size=width)
        self.w_output_scale=w_output_scale
        self.w_input_scale=w_input_scale
        self.primary_amount=primary_amount
        self.popsize=100
        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

        # 设置随机数种子
        setup_seed(seed)

        self.device = torch.device(device)

        if delay_device is not None:
            self.delay_device = torch.device(delay_device)
        else:
            self.delay_device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        self.sumspikes=[]
        self.sumspikes.append(torch.zeros(self.n_offsprings,self.num_of_liquid_layer_neurons).to(device))
        self.sumspikes.append(torch.zeros(self.n_offsprings,self.n_output).to(device))
        self.spiketime=[]
        self.spiketime.append(torch.zeros(self.n_offsprings,self.num_of_liquid_layer_neurons).to(device))
        self.spiketime.append(torch.zeros(self.n_offsprings,self.n_output).to(device))




        self.thre=[]
        self.thre.append(torch.zeros(self.n_offsprings,self.num_of_liquid_layer_neurons).to(device))
        self.thre.append(torch.zeros(self.n_offsprings,self.n_output).to(device))
        self.liquid_s_list = torch.zeros(self.n_offsprings,self.num_of_liquid_layer_neurons).to(device)

        # Input Layer------------------------------------------------------------------------------------------
        # random weights
        inputmatrix = np.zeros((self.n_offsprings,self.n_input, self.num_of_liquid_layer_neurons))
        row=np.zeros((self.n_offsprings,input_size,primary_amount),dtype=int)
        for i in range(input_size):
            row[:,i]=i
        halfliquid = [i for i in range(0,int(self.num_of_liquid_layer_neurons))]
        weight = np.ones([input_size, primary_amount])
        for i in range(self.n_offsprings):
            col=np.array(sample(halfliquid,primary_amount*self.n_input)).reshape(self.n_input,primary_amount)
            inputmatrix[i, row, col] = weight
        # col=np.random.randint(low=0,high=self.num_of_liquid_layer_neurons/2,size=(input_size,primary_amount))
        # weight=np.random.random(size=(input_size,primary_amount))



        self.input_to_primary_weight_matrix=torch.from_numpy(inputmatrix).to(self.device).float()
        # self.input_to_primary_weight_matrix=normalize(self.input_to_primary_weight_matrix,dim=1)
        self.input_to_primary_weight_matrix*=self.w_input_scale

        self.input_to_primary_list=update_matrix_to_list(self.input_to_primary_weight_matrix,id=self.num_of_liquid_layer_neurons)



        # Liquid Layer----------------------------------------------------------------------------------------
        # random weights
        self.liquid_weight_matrix=abs(torch.randn(size=(self.n_offsprings,self.num_of_liquid_layer_neurons, self.num_of_liquid_layer_neurons),device=self.device))
        # delete weights based on liquid density
        self.liquid_mask=torch.from_numpy(np.random.choice([0, 1], size=self.liquid_weight_matrix.size(), p=[liquid_density, 1-liquid_density])).to(device).bool()
        self.liquid_weight_matrix = self.liquid_weight_matrix.masked_fill(self.liquid_mask, 0)
        # symmetry
        # self.liquid_weight_matrix = (self.liquid_weight_matrix + self.liquid_weight_matrix.permute(0, 2, 1)) / 2

        # neuron distance
        dism = torch.from_numpy(self.priority).to(self.device).float()
        # # _, scale = torch.sort(dism, descending=True)
        self.liquid_weight_matrix*=dism
        self.liquid_weight_matrix=torch.triu(self.liquid_weight_matrix,diagonal=1)

        self.liquid_weight_matrix*=w_liquid_scale



        # output layer-----------------------------------------------------------------------------------------

        outputmatrix = np.zeros((self.n_offsprings,self.num_of_liquid_layer_neurons,self.n_output))
        output_mask_matrix = np.ones((self.n_offsprings,self.num_of_liquid_layer_neurons,self.n_output))
        row=np.zeros((secondary_amount,output_size),dtype=int)
        for i in range(output_size):
            row[:,i]=i
        liquid_to_output_list=[]

        halfliquid = [i for i in range(0,int(self.num_of_liquid_layer_neurons))]
        for i in range(self.n_offsprings):
            col=np.array(sample(halfliquid,secondary_amount*self.n_output)).reshape(secondary_amount,output_size) #对于每个agent，连接到output的5个液体层神经元编号：对应到每个output，5*out
            weight=np.random.random(size=(secondary_amount,output_size))
            outputmatrix[i,col,row]=weight
            output_mask_matrix[i,col,row]=0
        self.liquid_to_output_weight_matrix=torch.from_numpy(outputmatrix).to(self.device).float()
        self.liquid_to_output_list=update_matrix_to_list(self.liquid_to_output_weight_matrix.permute(0, 2, 1),id=self.num_of_liquid_layer_neurons+self.n_input)

        self.liquid_to_output_weight_matrix=regul(self.liquid_to_output_weight_matrix)
        self.liquid_to_output_weight_matrix*=w_output_scale
        self.readout_mask=torch.from_numpy(output_mask_matrix).to(self.device).bool()

    def predict_on_batch(self, input_state,i=-1,output='readout_values'):
        '''
            liquid_neuron: LIFNode, liquid layer neurons
            readout_neurons: LIFNode, readout layer neurons
            input_state: 4x1x3
            input_current: 4*64, input_state x primary
            primary_spikes: 0-1 matrix, 4*64, only primary
            liquid_current: 4*64, from primary to liquid
            liquid_spikes: 0-1 matrix, 4*64, liquid
            output_current: 4*64, from liquid to output
            readout_spikes: 0-1 matrix, 4*3, output
        '''
        if(torch.is_tensor(input_state)==False):
            input_state=torch.from_numpy(input_state).unsqueeze(1).to(device).float()

        input_state=input_state.reshape([input_state.size()[0],-1])
        input_current=torch.matmul(input_state.unsqueeze(1),self.input_to_primary_weight_matrix).squeeze()
        liquid_neurons = neuron.LIFNode(v_threshold=1.0)
        test_neurons = neuron.LIFNode(v_threshold=1.0)
        readout_neurons= neuron.LIFNode(v_threshold=1.0)
        liquid_neurons.reset()
        readout_neurons.reset()
        T = 10
        liquid_s_list = [] # 多秒放电的记录，time*off*64
        liquid_v_list = []
        out_s_list = []
        out_v_list = []
        for t in range(1,T):
            #####input to primary to output
            current1=input_current
            liquid_spikes1 = liquid_neurons(current1)
            liquid_v1=liquid_neurons.v
            output_current1=torch.matmul(liquid_spikes1.unsqueeze(dim=-2),self.liquid_to_output_weight_matrix.float()).squeeze() # liquid to output
            readout_spikes1 = readout_neurons(output_current1)
            readout_v1=readout_neurons.v

            #####primary to liquid to output
            current2=torch.matmul(liquid_spikes1.unsqueeze(1).float(),self.liquid_weight_matrix.float()).squeeze() # primary to liquid

            liquid_spikes2 = liquid_neurons(current2)
            liquid_v2=liquid_neurons.v
            output_current2=torch.matmul(liquid_spikes2.unsqueeze(dim=-2),self.liquid_to_output_weight_matrix.float()).squeeze() # liquid to output
            readout_spikes2 = readout_neurons(output_current2)
            readout_v2=readout_neurons.v

            #####liquid to liquid to output

            current3=torch.matmul(liquid_spikes2.unsqueeze(dim=-2),self.liquid_weight_matrix.float()).squeeze() # liquid to output

            liquid_spikes3 = liquid_neurons(current3)
            liquid_v3=liquid_neurons.v
            output_current3=torch.matmul(liquid_spikes3.unsqueeze(dim=-2),self.liquid_to_output_weight_matrix.float()).squeeze() # liquid to output
            readout_spikes3 = readout_neurons(output_current3)
            readout_v3=readout_neurons.v



            readout_spikes=readout_spikes1+readout_spikes2+readout_spikes3
            liquid_spikes=liquid_spikes1+liquid_spikes2+liquid_spikes3
            condi0=(self.spiketime[0]>0)&(self.spiketime[0]<t*liquid_spikes)
            condi1=(self.spiketime[1]>0)&(self.spiketime[1]<t*readout_spikes)

            self.spiketime[0]=torch.where(condi0,self.spiketime[0],t*liquid_spikes)
            self.spiketime[1]=torch.where(condi1,self.spiketime[1],t*readout_spikes)

            out_s_list.append(readout_spikes.cpu().numpy())
            out_v_list.append(readout_neurons.v.cpu().numpy())
            liquid_s_list.append(liquid_spikes.cpu().numpy())
            liquid_v_list.append(liquid_neurons.v.cpu().numpy())
            self.sumspikes[0] = 0.9 * self.sumspikes[0] + liquid_spikes
            a=liquid_spikes[0][6]
            b=self.sumspikes[0][0][6]
            self.thre[0] = torch.mean(self.sumspikes[0].float(), dim=1)
            self.thre[0] = torch.unsqueeze(self.thre[0], 1).repeat_interleave(repeats=self.num_of_liquid_layer_neurons,dim=1)
            self.sumspikes[1] = self.sumspikes[1] + readout_spikes
            self.thre[1] = torch.mean(self.sumspikes[1].float(), dim=1)
            self.thre[1] = torch.unsqueeze(self.thre[1], 1).repeat_interleave(repeats=self.n_output, dim=1)
        liquid_v_list=np.asarray(liquid_v_list)
        liquid_s_list=torch.from_numpy(np.asarray(liquid_s_list))
        out_v_list=np.asarray(out_v_list)
        out_s_list=torch.from_numpy(np.asarray(out_s_list))

        self.liquid_s_list=torch.sum(liquid_s_list,dim=0)
        self.out_s_list=torch.sum(out_s_list,dim=0)

        # print("spikes:",self.sumspikes[1][1])
        self.out=torch.max(self.sumspikes[1],dim=1)[1]

        # print("out:",out)
        return self.out

    def evolve(self,e):
        priority = calc_priority_based_on_dis(num=self.num_of_liquid_layer_neurons, size=self.width,neighbors=self.width)
        indiv=self.liquid_weight_matrix[i]
        for i in range(self.n_offsprings):
            pop=population(indiv.repeat(self.popsize,1,1)).pop
            if random.random()>0.999:
                inl = (self.input_to_primary_weight_matrix[i] == 0).nonzero().squeeze().cpu().numpy()
                ii=sample(range(inl.shape[0]),1)
                x=inl[ii][0][0]
                y=inl[ii][0][1]
                self.input_to_primary_weight_matrix[i][x][y]=self.w_input_scale
            r=[]
            for p in pop:
                spike_matrix=p.liquid_s_list 
                ll = (spike_matrix[0] == 0).nonzero().squeeze().cpu().numpy().tolist()
                if type(ll)==list:
                    silent_neurons=len(ll)
                    if silent_neurons/self.num_of_liquid_layer_neurons>0.1:
                        k = sample(ll, 1)[0]  # the chosen dead neuron
                        recent_active = (spike_matrix[i].bool().int() * priority[k]).argmax()  # Index of recent active neurons
                        p.liquid_weight_matrix[i][k][recent_active] += 0.1
                        p.liquid_weight_matrix[i][recent_active][k] +=0.1
                r,append(compute_rank(p))
            best_individual=pop[np.argmax(np.array(r))]
            self.liquid_weight_matrix[i]=best_individual.liquid_weight_matrix



    def reset_readout_weights(self):
        output_density=0.1
        spike_matrix=self.sumspikes[0]
        ll=(spike_matrix > 0).int().unsqueeze(-1).to(device)
        self.readout_mask=torch.from_numpy(np.random.choice([0, 1], size=self.liquid_to_output_weight_matrix.size(), p=[1-output_density, output_density])).to(device).bool()
        self.readout_mask=ll * self.readout_mask
        self.liquid_to_output_weight_matrix=self.readout_mask.float()
        self.liquid_to_output_list=update_matrix_to_list(self.liquid_to_output_weight_matrix.permute(0, 2, 1),id=self.num_of_liquid_layer_neurons+self.n_input)

    def reset(self):
        self.sumspikes = []
        self.sumspikes.append(
            torch.zeros(self.n_offsprings, self.num_of_liquid_layer_neurons).to(device))
        self.sumspikes.append(torch.zeros(self.n_offsprings, self.n_output).to(device))
        self.thre = []
        self.thre.append(torch.zeros(self.n_offsprings, self.num_of_liquid_layer_neurons).to(device))
        self.thre.append(torch.zeros(self.n_offsprings, self.n_output).to(device))


