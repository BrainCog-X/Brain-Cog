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
from BrainCog.base.strategy.surrogate import *

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random

from BrainCog.base.node.node import *
from BrainCog.base.learningrule.STDP import MutliInputSTDP

class CustomLinear(nn.Module):
    def __init__(self, weight,mask=None):
        super().__init__()

        self.weight = nn.Parameter(weight, requires_grad=True)
        self.mask=mask
    def forward(self, x: torch.Tensor):
        #
        # ret.shape = [C]
        return x.mul(self.weight) # Changed

    def update(self, dw):
        with torch.no_grad():
            if self.mask is not None:
                dw *= self.mask
            self.weight.data+= dw


class DLPFCNet(nn.Module):
    def __init__(self,connection):
        super().__init__()
        # DLPFC, BG     
        self.node = []
        self.node.append(IzhNodeMU(threshold=30., a=0.02, b=0.60, c=-65., d=8., mem=-70.)) 
        self.node.append(IzhNodeMU(threshold=30., a=0.02, b=0.60, c=-65., d=8., mem=-70.)) 

        self.learning_rule = []
        self.connection = connection

        self.out_DLPFC=torch.zeros((self.connection[0].weight.shape[1]), dtype=torch.float) # Input-DLPFC
        self.out_BG=torch.zeros((self.connection[1].weight.shape[1]), dtype=torch.float) # DLPFC-BG

    def forward(self, input): 
        self.out_DLPFC=self.node[0](self.connection[0](input))   
        self.out_BG=self.node[1](self.connection[1](self.out_DLPFC))

        BG_Spike = self.node[1].spike

        if sum(sum(BG_Spike)).item() > 1:
            num_neuron = len(BG_Spike)
            BG_Spike_index = torch.argmax(BG_Spike)
            BG_Spike_index_x = torch.floor(BG_Spike_index/num_neuron) 
            BG_Spike_index_y = BG_Spike_index - BG_Spike_index_x*num_neuron 

            BG_Spike = torch.zeros([num_neuron, num_neuron], dtype=torch.float) 
            BG_Spike[BG_Spike_index_x.long()][BG_Spike_index_y.long()] = 1 
        return BG_Spike

    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()

    def UpdateWeight(self, i, W):
        self.connection[i].weight.data = W


class OFCNet(nn.Module):
    def __init__(self,connection):
        super().__init__()
        # OFC, MOFC, LOFC    
        self.node = []
        self.node.append(IzhNodeMU(threshold=30., a=0.02, b=0.60, c=-65., d=8., mem=-70.)) # OFC_1 
        self.node.append(IzhNodeMU(threshold=30., a=0.02, b=0.60, c=-65., d=8., mem=-70.)) # OFC_2 
        self.node.append(IzhNodeMU(threshold=30., a=0.02, b=0.60, c=-65., d=8., mem=-70.)) # MOFC
        self.node.append(IzhNodeMU(threshold=30., a=0.02, b=0.60, c=-65., d=8., mem=-70.)) # LOFC
        
        self.connection = connection
        self.learning_rule = []

        self.learning_rule.append(MutliInputSTDP(self.node[3], [self.connection[3],self.connection[4]])) # OFC_2-LOFC, MOFC-LOFC
        self.learning_rule.append(MutliInputSTDP(self.node[3], [self.connection[3],self.connection[5]])) # OFC_2-LOFC, OFC_1-LOFC

        self.out_OFC_1=torch.zeros((self.connection[0].weight.shape[1]), dtype=torch.float) 
        self.out_OFC_2=torch.zeros((self.connection[1].weight.shape[1]), dtype=torch.float)
        self.out_MOFC=torch.zeros((self.connection[2].weight.shape[1]), dtype=torch.float)
        self.out_LOFC=torch.zeros((self.connection[5].weight.shape[1]), dtype=torch.float)
        

    def forward(self, Input_Tha, Input_SNc, Reward): 
        self.out_OFC_1 = self.node[0](self.connection[0](Input_Tha))
        self.out_OFC_2 = self.node[1](self.connection[1](Input_SNc))
        if Reward == 1:
            self.out_MOFC = self.node[2](self.connection[2](self.out_OFC_1))
            self.out_LOFC, dw_lofc = self.learning_rule[0](self.out_OFC_2, self.out_MOFC)
        else:
            self.out_MOFC = self.node[2](self.connection[2](self.out_OFC_1*0)) 
            self.out_LOFC, dw_lofc = self.learning_rule[1](self.out_OFC_2, self.out_OFC_1)
        
        MOFC_Spike = self.node[2].spike
        LOFC_Spike = self.node[3].spike

        return MOFC_Spike, LOFC_Spike
    
    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()


class BGNet(nn.Module):
    def __init__(self,connection):
        super().__init__()
        # DLPFC, StrD1, StrD2       
        self.node = []
        self.node.append(IzhNodeMU(threshold=30., a=0.02, b=0.60, c=-65., d=8., mem=-70.)) # DLPFC
        self.node.append(IzhNodeMU(threshold=30., a=0.01, b=0.01, c=-65., d=8., mem=-70.)) # StrD1 
        self.node.append(IzhNodeMU(threshold=30., a=0.1, b=0.5, c=-65., d=8., mem=-70.)) # StrD2
        
        self.connection = connection
        self.learning_rule = []
        
        self.out_DLPFC=torch.zeros((self.connection[0].weight.shape[1]), dtype=torch.float)
        self.out_StrD1=torch.zeros((self.connection[1].weight.shape[1]), dtype=torch.float)
        self.out_StrD2=torch.zeros((self.connection[2].weight.shape[1]), dtype=torch.float)


    def forward(self, input1, input2, input3): 
        self.out_DLPFC=self.node[0](self.connection[0](input1))       
        self.out_StrD1=self.node[1](self.connection[1](input2))
        self.out_StrD2=self.node[2](self.connection[2](input3))

        DLPFC_out = self.node[0].spike
        BG_out = self.node[1].spike + self.node[2].spike

        return DLPFC_out, BG_out

    def reset(self):
        for i in range(len(self.node)):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()
    def UpdateWeight(self, i, W):
        self.connection[i].weight.data = W


def STDP(Pre_mat, Post_mat, W):  
    T_Pre = 0
    T_Post = 0
    for i in range(len(Pre_mat)):       
        C_Pre = Pre_mat[i]
        C_Post = Post_mat[i]
        if sum(sum(C_Pre)) > 0:
            T_Pre = i
            Spike_Pre = Pre_mat[T_Pre]
        if sum(sum(C_Post)) > 0:
            T_Post = i
            Spike_Post = Post_mat[T_Post]
        if T_Pre*T_Post > 0:
            dT = T_Pre - T_Post   
    
            A_up = 0.777
            A_down = -0.237
            tau_up = 16.8
            tau_down = -33.7
            if dT < 0:
                dW = A_up * math.exp(dT/tau_up)
            else:
                dW = A_down * math.exp(dT/tau_down)           
            T_Post = 0         
            dW_mat = torch.mul(Spike_Pre, Spike_Post)*dW
            W = W + torch.mul(dW_mat, W)
    return W
              
    
if __name__=="__main__":
    # number of neurons
    num_neuron = 6
    num_DLPFC = num_neuron 
    num_BG  = num_neuron
    num_StrD1 = num_neuron
    num_StrD2 = num_neuron
    num_Thalamus = num_neuron
    num_OFC = num_neuron
    num_SNc = num_neuron
    num_PMC = num_neuron


    ##############################
    # DLPFC
    ##############################
    WeightAdd = 20
    # DLPFC-BG
    DLPFC_BG_connection = []
    # Input-DLPFC
    con_matrix0 = torch.ones([num_DLPFC, num_DLPFC], dtype=torch.float)*WeightAdd
    DLPFC_BG_connection.append(CustomLinear(con_matrix0))
    # DLPFC-BG
    W = torch.ones([num_DLPFC, num_BG], dtype=torch.float)*WeightAdd
    DLPFC_BG_connection.append(CustomLinear(W))

    DLPFC = DLPFCNet(DLPFC_BG_connection)


    ##############################
    # OFC
    ##############################
    WeightAdd = 20
    OFC_connection = []
    # Tha-OFC_1 (Input1)  
    con_matrix0 = torch.ones([num_Thalamus, num_OFC], dtype=torch.float)*WeightAdd
    OFC_connection.append(CustomLinear(con_matrix0))   
    # SNc/VTA-OFC_2 (Input2)
    con_matrix1 = torch.ones([num_SNc, num_OFC], dtype=torch.float)*WeightAdd
    OFC_connection.append(CustomLinear(con_matrix1))
    # OFC_1-MOFC
    con_matrix2 = torch.ones([num_OFC, num_OFC], dtype=torch.float)*WeightAdd*5
    OFC_connection.append(CustomLinear(con_matrix2)) 
    # OFC_2-LOFC
    con_matrix3 = torch.ones([num_OFC, num_OFC], dtype=torch.float)*WeightAdd*5
    OFC_connection.append(CustomLinear(con_matrix3)) 
    # MOFC-LOFC
    con_matrix4 = torch.ones([num_OFC, num_OFC], dtype=torch.float)*WeightAdd*-10
    OFC_connection.append(CustomLinear(con_matrix4)) 
    # OFC_1-LOFC
    con_matrix5 = torch.ones([num_OFC, num_OFC], dtype=torch.float)*WeightAdd*5
    OFC_connection.append(CustomLinear(con_matrix5)) 

    OFC = OFCNet(OFC_connection)


    ##############################
    # BGNet
    ##############################
    BG_connection = []
    WeightAdd = 20
    # Input1-DLPFC
    con_matrix0 = torch.ones([num_DLPFC, num_DLPFC], dtype=torch.float)*WeightAdd
    BG_connection.append(CustomLinear(con_matrix0))
    # Input2-StrD1
    con_matrix1 = torch.ones([num_StrD1,num_StrD1], dtype=torch.float)*WeightAdd
    BG_connection.append(CustomLinear(con_matrix1))
    # Input3-StrD2
    con_matrix2 = torch.ones([num_StrD2,num_StrD2], dtype=torch.float)*WeightAdd
    BG_connection.append(CustomLinear(con_matrix2))
    BG_connection.append(CustomLinear(W))
    # StrD1-BG
    con_matrix4 = torch.ones([num_StrD1,num_BG], dtype=torch.float)*WeightAdd
    BG_connection.append(CustomLinear(con_matrix4))
    # StrD2-BG
    con_matrix5 = torch.ones([num_StrD2,num_BG], dtype=torch.float)*WeightAdd
    BG_connection.append(CustomLinear(con_matrix5))

    BG = BGNet(BG_connection)


    ##############################
    # Train
    ##############################
    # Intention-action corresponding rules
    Intention_mat = range(num_neuron)
    Action_mat = range(num_neuron)

    TrainNum = 0
    for k in range(len(Intention_mat)):
        Intention = Intention_mat[k] 
        Intention_Action = Action_mat[k] 
        for j in range (len(Intention_mat)+1): 

            TrainNum = TrainNum + 1

            # Intention prediction
            for i in range(10):
                DLPFC_Input = torch.zeros([num_DLPFC, num_DLPFC], dtype=torch.float)
                DLPFC_Input[Intention,:] = 10
                BG_Spike = DLPFC(DLPFC_Input)
                if sum(sum(BG_Spike)).item() > 0:
                    Action = torch.nonzero(BG_Spike).numpy()[0][1]
                    break
            DLPFC.reset()
            
            PMC = torch.zeros([1, num_PMC], dtype=torch.float)
            PMC[0][Action] = 1

            Thalamus = torch.zeros([num_Thalamus, num_Thalamus], dtype=torch.float) 
            Thalamus[Intention][Action] = 10 
                  
            if Intention_Action == Action:
                # Positive reward
                # Tha-OFC_1-MOFC, SNc-OFC_2&MOFC-LOFC
                Reward = 1
                Input_Tha = Thalamus 
                Input_SNc_Reward = torch.ones([num_OFC, num_OFC], dtype=torch.float) 
                Input_SNc = torch.mul(Input_SNc_Reward, PMC)*10 
                for t in range(10):
                    MOFC_Spike, LOFC_Spike = OFC(Input_Tha, Input_SNc, Reward)
                
            else:
                # Negative reward
                # Tha-OFC_1-LOFC, SNc-OFC_2-LOFC (SNC is zeros)
                Reward = -1
                Input_Tha = Thalamus 
                Input_SNc = torch.zeros([num_OFC, num_OFC], dtype=torch.float) 
                for t in range(10):
                    MOFC_Spike, LOFC_Spike = OFC(Input_Tha, Input_SNc, Reward)
            OFC.reset()

            for i in range(1):
                DLPFC_out_mat = []
                BG_out_mat = []
                State = 0
                for t in range(10):   
                    DLPFC_Input = MOFC_Spike + LOFC_Spike
                    StrD1_Input = MOFC_Spike
                    StrD2_Input = LOFC_Spike

                    DLPFC_out, BG_out = BG(DLPFC_Input, StrD1_Input, StrD2_Input)
                    DLPFC_out_mat.append(DLPFC_out)
                    BG_out_mat.append(BG_out)
                
                W = STDP(DLPFC_out_mat, BG_out_mat, W)
                BG.reset()

                DLPFC.UpdateWeight(1, W)
                BG.UpdateWeight(3, W)

            if Reward == 1:
                break
    
    print("Train End")
    print("W is: \n", W)
    print("TrainNum is: \n", TrainNum)
    print("*****************************")