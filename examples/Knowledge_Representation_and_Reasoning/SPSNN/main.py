import time
import numpy as np
import os
import warnings
import math
from matplotlib import pyplot as plt
import torch
from braincog.base.node.node import *
from braincog.base.brainarea.BrainArea import *
from braincog.utils import *


warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)





class SPNet(BrainArea):
    """

    网络结构类:SPNet（Sequence Production Net)，定义了网络的结构，继承自BrainArea基类。
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param tau: 神经元膜电位常数，控制膜电位衰减
    :param decay:STDP机制衰减常数，控制STDP机制作用强度随时间变化
    :param w1:神经网络内部连接权重
    :param w2:外部输入电流到每个神经元的连接


    """

    def __init__(self, w1, w2 ):
        """
        """
        super().__init__()

        self.node = [LIFNode(threshold=10,tau=15) ]
        self.connection = [CustomLinear(w1), CustomLinear(w2) ]
        self.stdp = []

        self.stdp.append(MutliInputSTDP(self.node[0], [self.connection[0], self.connection[1]],decay=0.745))
        self.x1 = torch.zeros(1, w2.shape[0])

    def forward(self, x):
        """
        一次时间步的前向传播过冲函数，计算脉冲发放情况和权重改变量

        :param x1:经过该时间步后的脉冲发放情况
        :param dw1:STDP机制在一个时间步后带来的权重改变量
        """
        self.x1, dw1 = self.stdp[0]( self.x1,x)


        return self.x1, dw1

    def reset(self):
        self.x1 *= 0



def S_bound(S):
    """
    S_bound:网络权重边界控制函数，主要功能为控制全网络突触连接权不超过阈值，维持弱连接
    另外，本函数还需要将神经元组内部权重控制在一定的范围之内，以防神经元组不断重复激活自身的情况发生。

    :param synapse_bound: 全网络的突触连接的阈值，以维持网络整体为弱连接
    :param inner_bound: 神经元组内部突触连接的阈值，防止神经元组不断重复激活自身导致网络放电紊乱

    """

    S[S > synapse_bound]  =  synapse_bound
    S[S < -synapse_bound] = -synapse_bound


    temp1 = S[l1_stimu,:]
    temp2 = temp1[:, l1_stimu]
    temp2 [temp2>inner_bound] = inner_bound
    temp1[:, l1_stimu] = temp2
    S[l1_stimu, :] = temp1

    temp1 = S[l2_stimu, :]
    temp2 = temp1[:, l2_stimu]
    temp2[temp2 > inner_bound] = inner_bound
    temp1[:, l2_stimu] = temp2
    S[l2_stimu, :] = temp1

    temp1 = S[l3_stimu, :]
    temp2 = temp1[:, l3_stimu]
    temp2[temp2 > inner_bound] = inner_bound
    temp1[:, l3_stimu] = temp2
    S[l3_stimu, :] = temp1

    temp1 = S[l4_stimu, :]
    temp2 = temp1[:, l4_stimu]
    temp2[temp2 > inner_bound] = inner_bound
    temp1[:, l4_stimu] = temp2
    S[l4_stimu, :] = temp1


    return S


if __name__ == "__main__":



    ## Neurons Parameter

    C = 50;  # constant:the number of neurons of a symbol
    runtime = 1000;  # Runtime in ms


    thresh = 30;  # Judge if the neurons fire or not


    I_syn = 5;
    tau_m = 30;
    Rm = 10;
    learning_times = 3;
    Sym_size = 6
    I_t = 5  # Time duration of stimu current
    I_P = 130  # Strength of input current
    A_P = 0.024
    certainty = 0.35

    synapse_bound = 10  # The bound of all synapse
    inner_bound = 1  # The bound of population inner synapse

    """
        SPSNN主函数，实现网络核心主要功能

        :param C: 神经元组中神经元数量
        :param runtime: 网络总体模拟的时间步长
        :param learning_times: 网络进行序列学习的次数
        :param I_t: 对网络添加外部输入电流的时间长度
        :param I_P: 对网络添加外部输入电流的强度
        :param A_P: 网络在进行STDP学习后突触改变放缩量
        :param certainty: 网络输入电流大小的确定度  
        :param total_neurons: 网络神经元总量
        :param ADJ: 网络中脉冲放电情况矩阵
        :param I_stimu: 网络中外部输入电流矩阵
        :param S: 网络突触连接权重矩阵
        :param E: 单位矩阵，用以对每个神经元引入外部电流  

    """

    # Initial Neurual Network

    Net1 = [C,Sym_size*C,Sym_size*C,Sym_size*C,1]       #memory
    Net2 = [Sym_size,Sym_size,Sym_size]                 #action


    current_end   = 0
    total_neurons = int(sum(Net1)+sum(Net2))

    index1 = np.linspace(1,sum(Net1),sum(Net1),dtype=int)-1

    index2 = np.linspace (sum(Net1)+1,total_neurons,sum(Net2),dtype=int)-1


    Ne = total_neurons



    firings= []                       # spike timings


    ADJ    = np.zeros((runtime,Ne))   # record the firing condition
    abs_Ne = np.zeros(Ne)             # maintain the ABS of every neurons



    I_stimu = np.zeros((Ne,runtime));

    P       = np.zeros((runtime,3));   # potential of neuron 1



    # logical vector to differ if the neuron is belong to memory part
    If_Memory = np.zeros((total_neurons),dtype=bool)
    If_Memory[index1[:]] = True



    # logical vector to differ if the neuron is belong to action part

    If_Action = np.zeros((total_neurons),dtype=bool)
    If_Action[index2[:]] = True





    # Pre-set synapses

    S = np.zeros((total_neurons,total_neurons),dtype=float)           #  Initial Weights

    S = S - np.diag(S)      # Set the diag num to


    E = np.identity((total_neurons),dtype=float)



    W_r2a = 0.3


    # Memory to Action
    for i in range (C,sum(Net1)-2 ):
        S[ int(index2[int(i/C)-1]), i] =W_r2a




    # Learning Process
    I_stimu = np.zeros((Ne,runtime))
    seq = np.array([6,3,4])

    l1_stimu = np.arange  (0,C)
    l2_stimu = np.arange(C+(seq[0]-1)*C,C+(seq[0])*C)
    l3_stimu = np.arange(C+ 6*C+(seq[1]-1)*C,C+ 6*C+(seq[1])*C)
    l4_stimu = np.arange(C+12*C+(seq[2]-1)*C,C+12*C+(seq[2])*C)
    l5_stimu = Ne-1





    np.linspace(20 + i * 100 ,20+I_t+i*100-1,I_t ,dtype=int )

    for i in range(learning_times):
        """
        对网络添加输入电流
        """



        temp = I_stimu [l1_stimu,:]

        I_stimu [l1_stimu, 10 + i * 100 :10+I_t+i*100] = certainty*I_P + I_P * np.random.rand(C,I_t)
        I_stimu [l2_stimu, 25 + i * 100 :25+I_t+i*100] = certainty*I_P + I_P * np.random.rand(C,I_t)
        I_stimu [l3_stimu, 40 + i * 100 :40+I_t+i*100] = certainty*I_P + I_P * np.random.rand(C,I_t)
        I_stimu [l4_stimu, 55 + i * 100 :55+I_t+i*100] = certainty*I_P + I_P * np.random.rand(C,I_t)
        I_stimu [l5_stimu, 70 + i * 100 :70+I_t+i*100] = certainty*I_P + I_P * np.random.rand(1,I_t)



    I_stimu[l1_stimu,700:700+I_t] = I_P * np.random.rand(C,I_t)



    S = torch.tensor(S,dtype=torch.float32)
    E = torch.tensor(E,dtype=torch.float32)
    SPSNN = SPNet(S,E)




    for  t in range (runtime):

        I_input = torch.tensor( I_stimu[:,t].reshape(1,total_neurons),dtype=torch.float32)


        x,dw = SPSNN( I_input )

        S   += A_P*dw[1]

        S   += S_bound(S) - S



        ADJ[t] = x






plt.matshow(I_stimu)
plt.matshow(ADJ)

plt.matshow(S)
plt.colorbar()
plt.show()












