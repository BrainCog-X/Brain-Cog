import numpy as np
import torch,os,sys
from torch import nn
from torch.nn import Parameter 
import abc
import math
from abc import ABC
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from braincog.base.strategy.surrogate import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random
import gc
from braincog.base.node.node import IzhNodeMU
import objgraph
from pympler import tracker

class CustomLinear(nn.Module):
    def __init__(self, weight,mask=None):
        super().__init__()

        self.weight = nn.Parameter(weight, requires_grad=True)
        self.mask=mask
    def forward(self, x: torch.Tensor):
        #
        # ret.shape = [C]
        return x.mul(self.weight)

    def update(self, dw):
        with torch.no_grad():
            if self.mask is not None:
                dw *= self.mask
            self.weight.data+= dw

class M1Net(nn.Module):
    def __init__(self,connection):
        super().__init__()
        self.node = []
        self.node.append(IzhNodeMU(threshold=param_threshold, a=param_a, b=param_b, c=param_c, d=param_d, mem=param_mem, u=param_u, dt=param_dt))
        self.connection = connection

    def forward(self, input):        
        input_n = input*I_max
        input_r = torch.round(input_n)
        Spike = torch.zeros(num_neuron, dtype=torch.float)
        self.node[0].n_reset()
        if TrickID == 1:
            for i in range(num_AI):
                input_ri = int(input_r[i].item())
                FR_i = spike_num_list[input_ri]
                Spike[i] = FR_i
            FR_n = Spike
        else:
            for t in range(Simulation_time):    
                self.out=self.node[0](input_r)
                n_Spike = self.node[0].spike          
                Spike = Spike + n_Spike
            FR_n = Spike/Simulation_time
        return FR_n
    
    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()

class VNet(nn.Module):
    def __init__(self,connection):
        super().__init__()
        self.node = []
        self.node.append(IzhNodeMU(threshold=param_threshold, a=param_a, b=param_b, c=param_c, d=param_d, mem=param_mem, u=param_u, dt=param_dt))
        self.connection = connection

    def forward(self, input):        
        input_n = input*I_max
        input_r = torch.round(input_n)
        Spike = torch.zeros(num_neuron, dtype=torch.float)
        self.node[0].n_reset()
        if TrickID == 1:
            for i in range(num_neuron):
                input_ri = int(input_r[i].item())
                FR_i = spike_num_list[input_ri]
                Spike[i] = FR_i
            FR_n = Spike
        else:
            for t in range(Simulation_time):    
                self.out=self.node[0](input_r)
                n_Spike = self.node[0].spike          
                Spike = Spike + n_Spike
            FR_n = Spike/Simulation_time
        return FR_n

    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()

class S1Net(nn.Module):
    def __init__(self,connection):
        super().__init__()
        self.node = []
        self.node.append(IzhNodeMU(threshold=param_threshold, a=param_a, b=param_b, c=param_c, d=param_d, mem=param_mem, u=param_u, dt=param_dt))
        self.connection = connection

    def forward(self, input, FR, C, Fired, W_LatInh): 
        FR_W = torch.zeros(num_neuron, dtype=torch.float)
        if len(FR.shape) == 1:
            FR_W = FR*self.connection[0].weight
        else:
            for i in range(FR.shape[0]):
                FR_Wi = FR[i]*self.connection[i].weight
                FR_W = FR_W + FR_Wi
        sf = torch.tanh(FR_W)
        sf = torch.where(sf<0, 0, sf)
        input_n = -C * (input-sf) + input
        input_n = torch.where(input_n<0, 0, input_n)
        input = input_n*I_max
        input_r = torch.round(input)
        Spike = torch.zeros(num_S1, dtype=torch.float)
        self.node[0].n_reset()
        if TrickID == 1:
            for i in range(num_neuron):
                input_ri = int(input_r[i].item())
                FR_i = spike_num_list[input_ri]
                Spike[i] = FR_i
            FR_n = Spike
        else:
            for t in range(Simulation_time):    
                self.out=self.node[0](input_r)
                n_Spike = self.node[0].spike          
                Spike = Spike + n_Spike
            FR_n = Spike/Simulation_time

        S = input_n
        S = torch.where(input_n>= fire_threshold, 1, S)
        S = torch.where(input_n< fire_threshold, 0, S)
        if torch.sum(S) > 0:
            Fired = Fired + 1; 
            W_LatInh = torch.tanh(W_LatInh - 2 * torch.acos(S) * torch.exp(Fired) - 1) + 1

        return FR_n, input_n, Fired, W_LatInh

    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()

class EBANet(nn.Module):
    def __init__(self,connection):
        super().__init__()
        self.node = []
        self.node.append(IzhNodeMU(threshold=param_threshold, a=param_a, b=param_b, c=param_c, d=param_d, mem=param_mem, u=param_u, dt=param_dt))
        self.connection = connection

    def forward(self, input, FR, C, Fired, W_LatInh):  
        FR_W = torch.zeros(num_neuron, dtype=torch.float)
        if len(FR.shape) == 1:
            FR_W = FR*self.connection[0].weight
        else:
            for i in range(FR.shape[0]):
                FR_Wi = FR[i]*self.connection[i].weight
                FR_W = FR_W + FR_Wi
        sf = torch.tanh(FR_W)
        sf = torch.where(sf<0, 0, sf)
        input_n = -C * (input-sf) + input
        input_n = torch.where(input_n<0, 0, input_n)
        input = input_n*I_max
        input_r = torch.round(input)
        Spike = torch.zeros(num_S1, dtype=torch.float)
        self.node[0].n_reset()
        if TrickID == 1:
            for i in range(num_neuron):
                input_ri = int(input_r[i].item())
                FR_i = spike_num_list[input_ri]
                Spike[i] = FR_i
            FR_n = Spike
        else:
            for t in range(Simulation_time):    
                self.out=self.node[0](input_r)
                n_Spike = self.node[0].spike          
                Spike = Spike + n_Spike
            FR_n = Spike/Simulation_time
        
        S = input_n
        S = torch.where(input_n>= fire_threshold, 1, S)
        S = torch.where(input_n< fire_threshold, 0, S)
        if torch.sum(S) > 0:
            Fired = Fired + 1; 
            W_LatInh = torch.tanh(W_LatInh - 2 * torch.acos(S) * torch.exp(Fired) - 1) + 1

        return FR_n, input_n, Fired, W_LatInh

    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()

class TPJNet(nn.Module):
    def __init__(self,connection):
        super().__init__()
        self.node = []
        self.node.append(IzhNodeMU(threshold=param_threshold, a=param_a, b=param_b, c=param_c, d=param_d, mem=param_mem, u=param_u, dt=param_dt))
        self.connection = connection       
    
    def forward(self, input, FR, C):   
        FR_W = torch.zeros(num_neuron, dtype=torch.float)
        if len(FR.shape) == 1:
            FR_W = FR*self.connection[0].weight
        else:
            for i in range(FR.shape[0]):
                FR_Wi = FR[i]*self.connection[i].weight
                FR_W = FR_W + FR_Wi
        sf = torch.tanh(FR_W)
        sf = torch.where(sf<0, 0, sf)
        input_n = -C * (input-sf) + input
        input_n = torch.where(input_n<0, 0, input_n)
        input = input_n*I_max
        input_r = torch.round(input)
        Spike = torch.zeros(num_S1, dtype=torch.float)
        self.node[0].n_reset()
        if TrickID == 1:
            for i in range(num_neuron):
                input_ri = int(input_r[i].item())
                FR_i = spike_num_list[input_ri]
                Spike[i] = FR_i
            FR_n = Spike
        else:
            for t in range(Simulation_time):    
                self.out=self.node[0](input_r)
                n_Spike = self.node[0].spike          
                Spike = Spike + n_Spike
            FR_n = Spike/Simulation_time
        return FR_n, input_n

    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()

    def UpdateWeight(self, i, W):
        self.connection[i].weight.data = W

class AINet(nn.Module):
    def __init__(self,connection):
        super().__init__()
        self.node = []
        self.node.append(IzhNodeMU(threshold=param_threshold, a=param_a, b=param_b, c=param_c, d=param_d, mem=param_mem, u=param_u, dt=param_dt))
        self.connection = connection       

    def forward(self, input, FR, C):   
        FR_W = torch.zeros(num_neuron, dtype=torch.float)
        if len(FR.shape) == 1:
            FR_W = FR*self.connection[0].weight
        else:
            for i in range(FR.shape[0]):
                FR_Wi = FR[i]*self.connection[i].weight
                FR_W = FR_W + FR_Wi
        sf = torch.tanh(FR_W)
        sf = torch.where(sf<0, 0, sf)
        input_n = -C * (input-sf) + input
        input_n = torch.where(input_n<0, 0, input_n)
        input = input_n*I_max
        input_r = torch.round(input)
        Spike = torch.zeros(num_S1, dtype=torch.float)
        self.node[0].n_reset()
        if TrickID == 1:
            for i in range(num_neuron):
                input_ri = int(input_r[i].item())
                FR_i = spike_num_list[input_ri]
                Spike[i] = FR_i
            FR_n = Spike
        else:
            for t in range(Simulation_time):    
                self.out=self.node[0](input_r)
                n_Spike = self.node[0].spike          
                Spike = Spike + n_Spike
            FR_n = Spike/Simulation_time
        return FR_n, input_n
   
    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()

    def UpdateWeight(self, i, W, WIn):
        self.connection[i].weight.data = self.connection[i].weight.data + W*WIn


def DeltaWeight(Pre, Pre_n, Post, Post_n):
    alpha = -0.0035
    beta = 0.35
    gamma = -0.55
    T1 = alpha * (Pre_n*Post_n)
    T2 = beta * (Pre_n*(Post_n-Post))
    T3 = gamma * ((Pre_n-Pre)*Post_n)
    dW = T1 + T2 + T3
    return dW



if __name__=="__main__":
    """
    Set the number of neurons, and each neuron represents unique motion information (such as angle)
    """
    # number of neurons
    num_neuron = 9 
    num_M1 = num_neuron 
    num_S1  = num_neuron 
    num_TPJ = num_neuron
    num_V = num_neuron
    num_EBA = num_neuron
    num_AI = num_neuron

    Init_Weight = 1.

    param_threshold = 30.
    param_a = 0.02
    param_b = -0.1
    param_c = -55.
    param_d = 18.
    param_mem = -70.
    param_u = 0.
    param_dt = 1.
    Simulation_time = 1000
    I_max = 1000

    # When the TrickID is set to 1, it means that the mapping relationship from input current 
    # to firing rate is obtained directly by loading the Izh.npy, 
    # which can significantly reduce the program running time
    TrickID = 1 
    if TrickID == 1:
        spike_num_list=np.load('Izh.npy')
        spike_num_list = spike_num_list/I_max

    ##############################
    # M1
    ##############################
    # M1_Input-M1
    M1_connection = []
    con_matrix0 = torch.ones(num_M1, dtype=torch.float)*Init_Weight
    M1_connection.append(CustomLinear(con_matrix0))
    M1 = M1Net(M1_connection)
  
    ##############################
    # V
    ##############################
    # V_Input-V
    V_connection = []
    con_matrix3 = torch.ones(num_V, dtype=torch.float)*Init_Weight
    V_connection.append(CustomLinear(con_matrix3))
    V = VNet(V_connection)
 
    ##############################
    # S1
    ##############################
    # M1-S1
    S1_connection = []
    con_matrix1 = torch.ones(num_S1, dtype=torch.float)*Init_Weight
    S1_connection.append(CustomLinear(con_matrix1))
    S1 = S1Net(S1_connection)

    ##############################
    # EBA
    ##############################
    # V-EBA
    EBA_connection = []
    con_matrix4 = torch.ones(num_EBA, dtype=torch.float)*Init_Weight
    EBA_connection.append(CustomLinear(con_matrix4))
    EBA = EBANet(EBA_connection)

    ##############################
    # TPJ
    ##############################
    # S1-TPJ, EBA-TPJ
    TPJ_connection = []
    # S1-TPJ
    con_matrix2 = torch.ones(num_TPJ, dtype=torch.float)*Init_Weight*150
    TPJ_connection.append(CustomLinear(con_matrix2))
    # EBA-TPJ
    con_matrix5 = torch.ones(num_TPJ, dtype=torch.float)*Init_Weight*150
    TPJ_connection.append(CustomLinear(con_matrix5))
    TPJ = TPJNet(TPJ_connection)

    ##############################
    # AI
    ##############################
    # S1-AI, TPJ-AI, EBA-AI
    AI_connection = []
    # S1-AI
    con_matrix6 = torch.ones(num_AI, dtype=torch.float)*Init_Weight
    AI_connection.append(CustomLinear(con_matrix6))
    # TPJ-AI
    con_matrix7 = torch.ones(num_AI, dtype=torch.float)*Init_Weight
    AI_connection.append(CustomLinear(con_matrix7))
    # EBA-AI
    con_matrix8 = torch.ones(num_AI, dtype=torch.float)*Init_Weight
    AI_connection.append(CustomLinear(con_matrix8))
    AI = AINet(AI_connection)   
    
    ##############################
    # Coding
    ##############################
    S = 1
    ISI = 1
    JMax = int((num_neuron-1)/2)
    listJ = list(range(-JMax,JMax+1))
    Coding = torch.zeros([num_neuron, num_neuron], dtype=torch.float)
    for i in range(len(listJ)):
        e = float(listJ[i])
        listY = []
        for j in range(len(listJ)):
            x = float(listJ[j])
            y = math.exp(-(x-e)**2/(2*S**2))
            Coding[i][j] = y

    ##############################
    # Train
    ##############################
    MoveNum = 25  
    Time = 300
    CT = 100
    Motion_Start = 1
    Motion_End = Motion_Start + CT
    Vision_Start = Motion_End
    Vision_End = Vision_Start + CT

    CM1 = 0.04 
    CV = 0.04 
    CS1 = 0.04 
    CEBA = 0.04 
    CTPJ = 0.01 
    CAI = 0.15 

    for k in range(num_neuron):
        for i in range(MoveNum): 
            print(i)        
            M1_input = torch.zeros(num_M1, dtype=torch.float)
            V_input = torch.zeros(num_V, dtype=torch.float)
            S1_input = torch.zeros(num_S1, dtype=torch.float)
            TPJ_input = torch.zeros(num_TPJ, dtype=torch.float)
            EBA_input = torch.zeros(num_EBA, dtype=torch.float)
            AI_input = torch.zeros(num_AI, dtype=torch.float)           
            
            FR_M1 = torch.zeros(num_M1, dtype=torch.float)
            FR_V = torch.zeros(num_V, dtype=torch.float)
            FR_S1 = torch.zeros( num_S1, dtype=torch.float)
            FR_EBA = torch.zeros(num_EBA, dtype=torch.float)
            FR_TPJ = torch.zeros(num_TPJ, dtype=torch.float)
            FR_AI = torch.zeros( num_AI, dtype=torch.float)

            dW_S1TPJ = torch.zeros(num_M1, dtype=torch.float)
            dW_EBATPJ = torch.zeros(num_M1, dtype=torch.float)
            dW_S1AI = torch.zeros(num_M1, dtype=torch.float)
            dW_EBAAI = torch.zeros(num_M1, dtype=torch.float)

            fire_threshold = 0.7
            W_LatInh_Init = torch.ones(num_neuron, dtype=torch.float)*Init_Weight
            W_LatInh_S1_AI = W_LatInh_Init
            W_LatInh_EBA_AI = W_LatInh_Init
            Fired_S1 = torch.zeros(num_S1, dtype=torch.float)
            Fired_EBA = torch.zeros(num_EBA, dtype=torch.float)
            
            with torch.no_grad():
                for t in range(1,Time+1):
                    S_M1 = torch.zeros(num_M1, dtype=torch.float)
                    S_V = torch.zeros(num_V, dtype=torch.float)

                    if t>=Motion_Start and t<=Motion_End:
                        S_M1 = Coding[k]
                        M1_input = (1-(1-CM1)**t)*S_M1 
                    else:
                        M1_input = S_M1
                    
                        
                    if t>=Vision_Start and t<=Vision_End:
                        S_V = Coding[k]
                        V_input = (1-(1-CV)**(t-CT))*S_V 
                    else:
                        V_input = S_V

                    FR_M1_n = M1(M1_input)
                    
                    FR_V_n = V(V_input)       
                                
                    [FR_S1_n, S1_input_n, Fired_S1, W_LatInh_S1_AI] = S1(S1_input, FR_M1_n, CS1, Fired_S1, W_LatInh_S1_AI)

                    [FR_EBA_n, EBA_input_n, Fired_EBA, W_LatInh_EBA_AI] =  EBA(EBA_input, FR_V_n, CEBA, Fired_EBA, W_LatInh_EBA_AI)

                    FR_Input_TPJ_n = torch.stack((FR_S1_n, FR_EBA_n), 0)
                    [FR_TPJ_n, TPJ_input_n] = TPJ(TPJ_input, FR_Input_TPJ_n, CTPJ)
                    
                    FR_Input_AI = torch.stack((FR_S1_n, FR_TPJ_n, FR_EBA_n), 0)
                    [FR_AI_n, AI_input_n] = AI(AI_input, FR_Input_AI, CAI)

                    # Update weights
                    # S1-AI
                    ddW_S1AI = DeltaWeight(FR_S1, FR_S1_n, FR_AI, FR_AI_n)
                    dW_S1AI = dW_S1AI + ddW_S1AI
                    # EBA-AI
                    ddW_EBAAI = DeltaWeight(FR_EBA, FR_EBA_n, FR_AI, FR_AI_n)
                    dW_EBAAI = dW_EBAAI + ddW_EBAAI

                    FR_M1 = FR_M1_n
                    FR_V = FR_V_n
                    FR_S1 = FR_S1_n
                    FR_EBA = FR_EBA_n
                    FR_TPJ = FR_TPJ_n
                    FR_AI = FR_AI_n
                    

                    S1_input = S1_input_n
                    TPJ_input = TPJ_input_n
                    EBA_input = EBA_input_n
                    AI_input = AI_input_n

            AI.UpdateWeight(0, dW_S1AI, W_LatInh_S1_AI)
            AI.UpdateWeight(2, dW_EBAAI, W_LatInh_EBA_AI)
            
            print(AI.connection[0].weight.data) # dW_S1AI
            print(AI.connection[2].weight.data) # dW_EBAAI
            
            M1.reset()
            V.reset()
            S1.reset()
            EBA.reset()
            TPJ.reset()
            AI.reset()                
            
    np.save('W_S1_AI.npy', AI.connection[0].weight.data)
    np.save('W_EBA_AI.npy', AI.connection[2].weight.data)

    print('Training End')