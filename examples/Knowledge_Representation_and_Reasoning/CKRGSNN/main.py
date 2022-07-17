import time
import numpy as np
import os
import warnings
import scipy.io as scio
import math
from matplotlib import pyplot as plt
import torch
from BrainCog.base.node.node import *
import turicreate as tc
from BrainCog.base.brainarea.BrainArea import *
from BrainCog.utils import *


warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)


class CKRNet(BrainArea):
    """
    Commonsense Knowledge Representation  Net
    """

    def __init__(self, w1, w2):
        """
        """
        super().__init__()

        self.node = [LIFNode(threshold=16, tau=15)]
        self.connection = [CustomLinear(w1), CustomLinear(w2)]
        self.stdp = []

        self.stdp.append(MutliInputSTDP(self.node[0], [self.connection[0], self.connection[1]], decay=0.83))
        self.x1 = torch.zeros(1, w2.shape[0])

    def forward(self, x):
        """
        x is spike train
        """
        self.x1, dw1 = self.stdp[0](self.x1, x)

        return self.x1, dw1

    def reset(self):
        self.x1 *= 0


def S_bound(S):

    S[S > synapse_bound] = synapse_bound
    S[S < -synapse_bound] = -synapse_bound

    for i in range(N_entity):
        temp1 = S[Index_E[i], :]
        temp2 = temp1[:, Index_E[i]]
        temp2[temp2 > inner_bound_E] = inner_bound_E
        temp1[:, Index_E[i]] = temp2
        S[Index_E[i], :] = temp1

    for i in range(N_relation):
        temp1 = S[Index_R[i], :]
        temp2 = temp1[:, Index_R[i]]
        temp2[temp2 > inner_bound_R] = inner_bound_R
        temp1[:, Index_R[i]] = temp2
        S[Index_R[i], :] = temp1

    return S


if __name__ == "__main__":

    print(os.getcwd())

    KG = tc.SFrame.read_csv('./sub_Conceptnet.csv')

    Set_R = set()
    Set_E = set()

    for i in range(KG.shape[0]):

        Set_R.add(KG[i]['Relation'])
        Set_E.add(KG[i]['Head'])
        Set_E.add(KG[i]['Tail'])

    List_E = sorted(Set_E)
    List_R = list(Set_R)
    List_R.sort()

    # Network Parameter#dkenf.kejlklkelkvjlkxjel

    I_syn = 5
    tau_m = 30
    I_t = 3  # Time duration of stimu current
    I_P = 150  # Strength of input current
    A_P = 0.009
    certainty = 0.2

    synapse_bound = 1    # The bound of all synapse
    inner_bound_E = 0.6  # The bound of population inner synapse
    inner_bound_R = 0.3  # The bound of population inner synapse

    Ce = 20   # num of entity
    Cr = 100  # num of relation
    N_entity = len(List_E)
    N_relation = len(List_R)
    total_neurons = Ce * N_entity + Cr * N_relation

    KG_No = KG.shape[0]
    trail_time = 40
    runtime = KG_No * trail_time

    print('N_entity=', N_entity)
    print('N_relation=', N_relation)
    print('KG_No=', KG_No)
    print('runtime=', runtime)
    print('total_neurons=', total_neurons)

    S = np.zeros((total_neurons, total_neurons), dtype=float)  # Initial Weights
    S = torch.tensor(S, dtype=torch.float32)
    E = np.identity((total_neurons), dtype=float)
    E = torch.tensor(E, dtype=torch.float32)

    I_stimu = np.zeros((total_neurons, runtime))
    ADJ = np.zeros((total_neurons, runtime))  # record the firing condition

    Index_E = []
    Index_R = []
    for i in range(N_entity):
        Index_E.append(np.arange(i * Ce, i * Ce + Ce))

    for i in range(N_relation):
        Index_R.append(np.arange(N_entity * Ce + i * Cr, N_entity * Ce + i * Cr + Cr))

    for i in range(KG_No):
        Head = KG[i]['Head']
        Rela = KG[i]['Relation']
        Tail = KG[i]['Tail']
        Weig = KG[i]['Weight']

        # print(List_E.index(Head))
        # print(List_R.index(Rela))
        # print(List_E.index(Rela))
        # print(Index_R[List_R.index(Rela)])

        I_stimu[Index_E[List_E.index(Head)], 10 + i * trail_time: 10 + I_t + i * trail_time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)
        I_stimu[Index_R[List_R.index(Rela)], 15 + i * trail_time: 15 + I_t + i * trail_time] = certainty * I_P + I_P * np.random.rand(Cr, I_t)
        I_stimu[Index_E[List_E.index(Tail)], 20 + i * trail_time: 20 + I_t + i * trail_time] = certainty * I_P + I_P * np.random.rand(Ce, I_t)

    CKRGSNN = CRKNet(S, E)

    for t in range(runtime):

        I_input = torch.tensor(I_stimu[:, t].reshape(1, total_neurons), dtype=torch.float32)

        x, dw = CKRGSNN(I_input)

        S += A_P * dw[1]

        S += S_bound(S) - S

        ADJ[:, t] = x
        print(t, 'step in >>', runtime)

    img_I = plt.matshow(I_stimu)
    plt.savefig("I_stimu1.jpg", dpi=500, bbox_inches='tight')

    img_ADJ = plt.matshow(ADJ)
    plt.savefig("ADJ1.jpg", dpi=500, bbox_inches='tight')

    img_S = plt.matshow(S)
    plt.colorbar()
    plt.savefig("S1.jpg", dpi=500, bbox_inches='tight')

    plt.show()

    S = np.mat(S)
    dataNew = './data_save.mat'
    scio.savemat(dataNew, {'I_stimu': I_stimu, 'ADJ': ADJ, 'Weight': S})
