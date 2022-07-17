import time
import numpy as np
import os
import warnings
import math
from matplotlib import pyplot as plt
import torch
from BrainCog.base.node.node import *
from BrainCog.base.brainarea.BrainArea import *
from BrainCog.utils import *


warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)


class CRNet(BrainArea):
    """

       网络结构类:CRNet（Causal Reasoning Net)，定义了网络的结构，继承自BrainArea基类。
       :param threshold: 神经元发放脉冲需要达到的阈值
       :param tau: 神经元膜电位常数，控制膜电位衰减
       :param decay:STDP机制衰减常数，控制STDP机制作用强度随时间变化
       :param w1:神经网络内部连接权重
       :param w2:外部输入电流到每个神经元的连接

    """

    def __init__(self, w1, w2):
        """
        """
        super().__init__()

        self.node = [LIFNode(threshold=16, tau=15)]
        self.connection = [CustomLinear(w1), CustomLinear(w2)]
        self.stdp = []

        self.stdp.append(MutliInputSTDP(self.node[0], [self.connection[0], self.connection[1]], decay=0.8))
        self.x1 = torch.zeros(1, w2.shape[0])

    def forward(self, x):
        """
        一次时间步的前向传播过冲函数，计算脉冲发放情况和权重改变量

        :param x1:经过该时间步后的脉冲发放情况
        :param dw1:STDP机制在一个时间步后带来的权重改变量
        """
        self.x1, dw1 = self.stdp[0](x, self.x1)

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

    S[S > synapse_bound] = synapse_bound
    S[S < -synapse_bound] = -synapse_bound

    temp1 = S[E1_index, :]
    temp2 = temp1[:, E1_index]
    temp2[temp2 > inner_bound_E] = inner_bound_E
    temp1[:, E1_index] = temp2
    S[E1_index, :] = temp1

    temp1 = S[E2_index, :]
    temp2 = temp1[:, E2_index]
    temp2[temp2 > inner_bound_E] = inner_bound_E
    temp1[:, E2_index] = temp2
    S[E2_index, :] = temp1

    temp1 = S[E3_index, :]
    temp2 = temp1[:, E3_index]
    temp2[temp2 > inner_bound_E] = inner_bound_E
    temp1[:, E3_index] = temp2
    S[E3_index, :] = temp1

    temp1 = S[E4_index, :]
    temp2 = temp1[:, E4_index]
    temp2[temp2 > inner_bound_E] = inner_bound_E
    temp1[:, E4_index] = temp2
    S[E4_index, :] = temp1

    temp1 = S[E4_index, :]
    temp2 = temp1[:, E4_index]
    temp2[temp2 > inner_bound_E] = inner_bound_E
    temp1[:, E4_index] = temp2
    S[E4_index, :] = temp1

    temp1 = S[E5_index, :]
    temp2 = temp1[:, E5_index]
    temp2[temp2 > inner_bound_E] = inner_bound_E
    temp1[:, E5_index] = temp2
    S[E5_index, :] = temp1

    temp1 = S[R1_index, :]
    temp2 = temp1[:, R1_index]
    temp2[temp2 > inner_bound_R] = inner_bound_R
    temp1[:, R1_index] = temp2
    S[R1_index, :] = temp1

    temp1 = S[R2_index, :]
    temp2 = temp1[:, R2_index]
    temp2[temp2 > inner_bound_R] = inner_bound_R
    temp1[:, R2_index] = temp2
    S[R2_index, :] = temp1

    return S


if __name__ == "__main__":

    # Neurons Parameter

    Cr = 200   # num of relation
    Ce = 50    # num of entity

    total_time = 2500            # Runtime in ms

    tau = 100             # time constant of STDP
    stdpwin = 25               # STDP windows in ms
    thresh = 30               # Judge if the neurons fire or not
    abs_T = 25               # The length of the ABS
    Reset = 0                # Reset Potential
    I_syn = 5
    tau_m = 30
    Rm = 10

    N_entity = 5
    N_relation = 2
    I_t = 5          # Duration of Current
    I_P = 25         # Strength of input current
    certainty = 0.5

    A_P = 0.01
    synapse_bound = 0.2   # The bound of all synapse
    inner_bound_E = 0.08   # The bound of population inner synapse
    inner_bound_R = 0.06   # The bound of population inner synapse

    total_neurons = Ce * N_entity + Cr * N_relation

    """
        SPSNN主函数，实现网络核心主要功能

        :param Cr: 因果图中节点神经元组中神经元数量
        :param Ce: 因果图中因果关系神经元组中神经元数量
        :param total_time: 网络总体模拟的时间步长
        :param learning_times: 网络进行序列学习的次数
        :param N_entity: 对网络添加外部输入电流的时间长度
        :param N_relation: 对网络添加外部输入电流的强度
        :param A_P: 网络在进行STDP学习后突触改变放缩量
        :param certainty: 网络输入电流大小的确定度
        :param total_neurons: 网络神经元总量
        :param ADJ: 网络中脉冲放电情况矩阵
        :param I_stimu: 网络中外部输入电流矩阵
        :param S: 网络突触连接权重矩阵
        :param E: 单位矩阵，用以对每个神经元引入外部电流

    """

    # Initial Neurual Network

    E1_index = np.linspace(0, Ce - 1, Ce, dtype=int)
    E2_index = np.linspace(Ce, 2 * Ce - 1, Ce, dtype=int)
    E3_index = np.linspace(2 * Ce, 3 * Ce - 1, Ce, dtype=int)
    E4_index = np.linspace(3 * Ce, 4 * Ce - 1, Ce, dtype=int)
    E5_index = np.linspace(4 * Ce, 5 * Ce - 1, Ce, dtype=int)

    R1_index = np.linspace(5 * Ce, 5 * Ce + Cr - 1, Cr, dtype=int)
    R2_index = np.linspace(5 * Ce + Cr, 5 * Ce + 2 * Cr - 1, Cr, dtype=int)

    Ne = total_neurons

    v = Reset * np.zeros(Ne)

    firings = []                       # spike timings

    ADJ = np.zeros((total_time, Ne))   # record the firing condition
    abs_Ne = np.zeros(Ne)             # maintain the ABS of every neurons

    I_stimu = np.zeros((Ne, total_time), dtype=float)

    If_Memory = np.zeros((total_neurons), dtype=bool)
    If_Memory[:] = True

    # Pre-set synapses

    S = np.zeros((total_neurons, total_neurons), dtype=float)  # Initial Weights

    S = S - np.diag(S)      # Set the diag num to

    E = np.identity((total_neurons), dtype=float)

    W_set_innner = 0.7

    temp = S[E1_index, :]
    temp[:, E1_index] = W_set_innner * np.random.rand(Ce, Ce)
    S[E1_index, :] = temp

    temp = S[E2_index, :]
    temp[:, E2_index] = W_set_innner * np.random.rand(Ce, Ce)
    S[E2_index, :] = temp

    temp = S[E3_index, :]
    temp[:, E3_index] = W_set_innner * np.random.rand(Ce, Ce)
    S[E3_index, :] = temp

    temp = S[E4_index, :]
    temp[:, E4_index] = W_set_innner * np.random.rand(Ce, Ce)
    S[E4_index, :] = temp

    temp = S[E5_index, :]
    temp[:, E5_index] = W_set_innner * np.random.rand(Ce, Ce)
    S[E5_index, :] = temp

    temp = S[R1_index, :]
    temp[:, R1_index] = 0.25 * W_set_innner * np.random.rand(Cr, Cr)
    S[R1_index, :] = temp

    temp = S[R2_index, :]
    temp[:, R2_index] = 0.25 * W_set_innner * np.random.rand(Cr, Cr)
    S[R2_index, :] = temp

    """
    对于因果图中的因果关系，给予网络不同神经元组输入电流刺激，使其建立连接
    """
    i = 1
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E1_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E1_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R1_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R1_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E2_index, :] = temp

    i = 3
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E2_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R2_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E1_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E1_index, :] = temp

    i = 6
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E2_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R1_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R1_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E3_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E3_index, :] = temp

    i = 8
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E3_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E3_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R2_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E2_index, :] = temp

    i = 11
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E2_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R1_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R1_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E4_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E4_index, :] = temp

    i = 13
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E4_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E4_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R2_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E2_index, :] = temp

    i = 16
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E3_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E3_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R1_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R1_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E5_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E5_index, :] = temp

    i = 18
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E5_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E5_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R2_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E3_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E3_index, :] = temp

    i = 21
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E4_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E4_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R1_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R1_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E5_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E5_index, :] = temp

    i = 23
    time = np.linspace(11 + i * 100, 10 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E5_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E5_index, :] = temp

    time = np.linspace(21 + i * 100, 20 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[R2_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
    I_stimu[R2_index, :] = temp

    time = np.linspace(31 + i * 100, 30 + I_t + i * 100, I_t, dtype=int)
    temp = I_stimu[E4_index, :]
    temp[:, time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
    I_stimu[E4_index, :] = temp

    S = torch.tensor(S, dtype=torch.float32)
    E = torch.tensor(E, dtype=torch.float32)
    CRSNN = CRNet(S, E)

    for t in range(total_time):
        I_input = torch.tensor(I_stimu[:, t].reshape(1, total_neurons), dtype=torch.float32)

        x, dw = CRSNN(I_input)
        S += A_P * dw[1]

        S += S_bound(S) - S

        ADJ[t] = x

    plt.matshow(I_stimu)
    plt.matshow(ADJ.transpose())

    plt.matshow(S)
    plt.colorbar()

    plt.show()
