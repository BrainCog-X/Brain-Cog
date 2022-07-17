from BrainCog.base.brainarea.Insula import *
from BrainCog.base.brainarea.IPL import *
from BrainCog.base.learningrule.STDP import *
from BrainCog.base.node.node import *
from BrainCog.base.connection.CustomLinear import *
import random
import numpy as np
import torch
import os
import sys
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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    """
    Set the number of neurons, and each neuron represents unique motion information (such as angle)
    """
    # number of neurons
    num_neuron = 5
    num_vPMC = num_neuron
    num_STS = num_neuron
    num_IPLM = num_neuron
    num_IPLV = num_neuron
    num_Insula = num_neuron

    """
    Setting the network structure and the initial weight of IPL
    """
    # IPLNet
    # connection
    connection = []
    # vPMC-IPLM
    con_matrix0 = torch.eye(num_IPLM, dtype=torch.float) * 2.5
    connection.append(CustomLinear(con_matrix0))
    # STS-IPLV
    con_matrix1 = torch.eye(num_IPLV, dtype=torch.float) * 2.5
    connection.append(CustomLinear(con_matrix1))
    # IPLM-IPLV
    con_matrix2 = torch.zeros([num_IPLM, num_IPLV], dtype=torch.float)
    connection.append(CustomLinear(con_matrix2))

    IPL = IPLNet(connection)

    print("IPL Connection (Before training):", connection[2].weight)

    """
    Setting the network structure and the initial weight of Insula
    """
    # InsulaNet
    # connection
    Insula_connection = []
    # IPLV-Insula
    con_matrix0 = torch.eye(num_IPLM, dtype=torch.float) * 2
    Insula_connection.append(CustomLinear(con_matrix0))
    # STS-Insula
    con_matrix1 = torch.eye(num_IPLV, dtype=torch.float) * 2
    Insula_connection.append(CustomLinear(con_matrix1))

    Insula = InsulaNet(Insula_connection)

    """
    Training process
    :param train_num: number of movements during training
    """
    # Train
    for vPMC_Angel in range(1, num_vPMC + 1):
        # vPMC Angle
        vPMC_Angel_v = torch.zeros([1, num_vPMC], dtype=torch.float)
        vPMC_Angel_v[0, vPMC_Angel - 1] = 20

        dwIPL_temp = torch.zeros([num_IPLM, num_IPLV], dtype=torch.float)

        train_num = 10
        for i_train in range(train_num):

            # STS 1
            STS_Angel_1 = vPMC_Angel
            for t in range(2):
                vPMC_input = vPMC_Angel_v
                STS_Angel_v = torch.zeros([1, num_STS], dtype=torch.float)
                STS_Angel_v[0, STS_Angel_1 - 1] = 20
                STS_input = STS_Angel_v
                IPLV_out, dwIPL = IPL(vPMC_input, STS_input)
                dwIPL_temp = dwIPL_temp + dwIPL
            IPL.reset()

            # STS 2
            STS_Angel_2 = random.randint(1, num_neuron)
            for t in range(2):
                vPMC_input = vPMC_Angel_v
                STS_Angel_v = torch.zeros([1, num_STS], dtype=torch.float)
                STS_Angel_v[0, STS_Angel_2 - 1] = 20
                STS_input = STS_Angel_v
                IPLV_out, dwIPL = IPL(vPMC_input, STS_input)
                dwIPL_temp = dwIPL_temp + dwIPL
            IPL.reset()

            # STS 3
            STS_Angel_3 = random.randint(1, num_neuron)
            for t in range(2):
                vPMC_input = vPMC_Angel_v
                STS_Angel_v = torch.zeros([1, num_STS], dtype=torch.float)
                STS_Angel_v[0, STS_Angel_3 - 1] = 20
                STS_input = STS_Angel_v
                IPLV_out, dwIPL = IPL(vPMC_input, STS_input)
                dwIPL_temp = dwIPL_temp + dwIPL
            IPL.reset()

        IPL.UpdateWeight(2, dwIPL_temp)

    print("IPL Connection (After training):", connection[2].weight)

    """
    Test process
    :param move_count: number of movements during test
    """
    # Test
    move_count = 10
    TestList_vPMC_Angel = np.random.randint(1, num_vPMC, move_count)
    TestList_STS_Angel_1 = TestList_vPMC_Angel
    TestList_STS_Angel_2 = np.random.randint(1, num_STS, move_count)
    TestList_STS_Angel_3 = np.random.randint(1, num_STS, move_count)
    TestMat_STS_Angle = np.vstack((TestList_STS_Angel_1, TestList_STS_Angel_2, TestList_STS_Angel_3))
    np.random.shuffle(TestMat_STS_Angle)

    TestList_IPLV_out = []
    for i_test in range(move_count):
        Test_vPMC_Angel = TestList_vPMC_Angel[i_test]
        Test_vPMC_Angel_v = torch.zeros([1, num_vPMC], dtype=torch.float)
        Test_vPMC_Angel_v[0, Test_vPMC_Angel - 1] = 20
        Test_STS_Angel_v = torch.zeros([1, num_STS], dtype=torch.float)
        for t in range(2):
            IPL(Test_vPMC_Angel_v, Test_STS_Angel_v)
            IPLV_out_f = torch.argmax(IPL.node[1].u) + 1
        IPL.reset()
        TestList_IPLV_out.append(IPLV_out_f.numpy().item())

    confidence = [0, 0, 0]
    for i in range(move_count):
        theta_predict = TestList_IPLV_out[i]
        theta_visual_1 = TestMat_STS_Angle[0][i]
        theta_visual_2 = TestMat_STS_Angle[1][i]
        theta_visual_3 = TestMat_STS_Angle[2][i]

        Test_IPL_v = torch.zeros([1, num_IPLV], dtype=torch.float)
        Test_IPL_v[0, theta_predict - 1] = 20

        Test_STS1_v = torch.zeros([1, num_STS], dtype=torch.float)
        Test_STS1_v[0, theta_visual_1 - 1] = 20
        for t in range(2):
            Insula(Test_IPL_v, Test_STS1_v)
        if sum(sum(Insula.out_Insula)) > 0:
            confidence[0] = confidence[0] + 1
        Insula.reset()

        Test_STS2_v = torch.zeros([1, num_STS], dtype=torch.float)
        Test_STS2_v[0, theta_visual_2 - 1] = 20
        for t in range(2):
            Insula(Test_IPL_v, Test_STS2_v)
        if sum(sum(Insula.out_Insula)) > 0:
            confidence[1] = confidence[1] + 1
        Insula.reset()

        Test_STS3_v = torch.zeros([1, num_STS], dtype=torch.float)
        Test_STS3_v[0, theta_visual_3 - 1] = 20
        for t in range(2):
            Insula(Test_IPL_v, Test_STS3_v)
        if sum(sum(Insula.out_Insula)) > 0:
            confidence[2] = confidence[2] + 1
        Insula.reset()

    x_0 = torch.arange(0, move_count)
    x_1 = torch.arange(move_count * 1, move_count * 2)
    x_2 = torch.arange(move_count * 2, move_count * 3)

    color_list = ['k', 'k', 'k']
    color_list[confidence.index(max(confidence))] = 'r'

    plt.subplot(211)
    plt.figure(1)
    plt.plot(x_0, TestMat_STS_Angle[0], color=color_list[0])
    plt.plot(x_1, TestMat_STS_Angle[1], color=color_list[1])
    plt.plot(x_2, TestMat_STS_Angle[2], color=color_list[2])
    plt.title("Motion Detection")
    plt.subplot(212)
    plt.plot(x_0, TestList_IPLV_out, color='r')
    plt.title("Motion Prediction")
    plt.tight_layout()
    plt.show()
