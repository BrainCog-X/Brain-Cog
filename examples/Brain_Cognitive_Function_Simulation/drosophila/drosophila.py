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
from braincog.base.node.node import IFNode
from braincog.base.learningrule.STDP import STDP,MutliInputSTDP
from braincog.base.connection.CustomLinear import CustomLinear
from braincog.model_zoo.nonlinearNet import droDMTestNet
from braincog.model_zoo.linearNet import droDMTrainNet
import copy

if __name__=="__main__":
    """
    建立训练网络
    """
    num_state=5
    num_action=2
    weight_exc=0.5
    weight_inh=-0.05
    trace_decay=0.8
    mb_connection=[]
    #input-visual
    con_matrix0 = torch.eye((num_state), dtype=torch.float)
    mb_connection.append(CustomLinear(weight_exc * con_matrix0,con_matrix0))
    # visual-kc
    con_matrix1 =torch.eye((num_state), dtype=torch.float)
    mb_connection.append(CustomLinear( weight_exc * con_matrix1,con_matrix1))
    # kc-mbon
    con_matrix2 = torch.ones((num_state,num_action), dtype=torch.float)
    mb_connection.append(CustomLinear(weight_exc * con_matrix2,con_matrix2))
    # mbon-mbon
    con_matrix3 = torch.ones((num_action,num_action), dtype=torch.float)
    con_matrix4 = torch.eye((num_action), dtype=torch.float)
    con_matrix5=con_matrix3-con_matrix4
    con_matrix5=con_matrix5
    mb_connection.append(CustomLinear(weight_inh * con_matrix5,con_matrix5))

    MB=droDMTrainNet(mb_connection)
    weight_trace_mbon=torch.zeros(con_matrix2.shape, dtype=torch.float)
    """
    学习绿色正立T是安全的  蓝色倒立T是有惩罚的
    """
    #learning GT
    # RGB T t
    GT = torch.tensor([0, 0.8, 0, 1.0, 0])
    Bt = torch.tensor([0, 0, 0.8, 0, 1.0])
    input = GT - Bt  # input GT
    input[input < 0] = 0
    for i_train in range(20):
        GT_out,dwkc,dwmbon=MB(input)
        print("stdp:",dwkc,dwmbon)
        #vis-kc STDP
        MB.UpdateWeight(1, dwkc)
        #kc-mbon rstdp
        weight_trace_mbon *= trace_decay
        weight_trace_mbon += dwmbon
        if max(GT_out)>0:
            r=torch.ones((num_state,num_action), dtype=torch.float)
            p_action= torch.tensor([0])
            r[:,p_action]=-1
            dw_mbon = r * weight_trace_mbon
            MB.UpdateWeight(2, dw_mbon)
            print("output:",GT_out)

    MB.reset()
    weight_trace_mbon = torch.zeros(con_matrix2.shape, dtype=torch.float)
    #learning Bt
    GT = torch.tensor([0,0.8,0, 1.0, 0])
    Bt = torch.tensor([0, 0, 0.8, 0, 1.0])
    input = Bt - GT  # input Bt
    input[input < 0] = 0
    for i_train in range(20):
        GT_out,dwkc,dwmbon=MB(input)
        #vis-kc STDP
        MB.UpdateWeight(1, dwkc)
        #kc-mbon rstdp
        weight_trace_mbon *= trace_decay
        weight_trace_mbon += dwmbon
        if max(GT_out)>0:
            r=torch.ones((num_state,num_action), dtype=torch.float)
            p_action= torch.tensor([1])
            r[:,p_action]=-1
            dw_mbon = r * weight_trace_mbon
            MB.UpdateWeight(2, dw_mbon)
    train_weight=MB.getweight()
    for i in range(len(train_weight)):
        print("weight after learning:", train_weight[i].weight.data)
    print("end training")


    #linear test conflict decision making
    test_num=12
    t1=torch.zeros((test_num), dtype=torch.float)
    t2=torch.zeros((test_num), dtype=torch.float)
    for c in range(t1.shape[0]):
        MB_test = droDMTrainNet(copy.deepcopy(train_weight))
        MB_test.reset()
        Gt = torch.tensor([0, (c*0.1), 0, 0, 0.5])
        BT = torch.tensor([0, 0, (c*0.1), 0.5, 0])
        input =Gt - BT   # input Gt
        input[input < 0] = 0
        count=torch.zeros((num_action), dtype=torch.float)
        for i_train in range(500):
            GT_out,dwkc,dwmbon=MB_test(input)
            count+=GT_out
        t1[c]=count[0]
        t2[c]=count[1]
    p1=(t1-t2)/(t1+t2)
    print(t1,t2,p1)
    for i in range(len(train_weight)):
        print("weight after learning:", train_weight[i].weight.data)

    """
    建立测试网络，验证不同浓度下绿色正立T和蓝色倒立T
    """
    # non-linear test conflict decision making
    weight_inh_test=-0.3
    num_apl=2
    num_da=1
    da_mb_connection=train_weight
    # kc-apl
    con_matrix6 = torch.ones((num_state, num_apl), dtype=torch.float)
    da_mb_connection.append(CustomLinear(weight_exc * con_matrix6, con_matrix6))
    # apl-kc
    con_matrix7 = torch.ones((num_apl,num_state), dtype=torch.float)
    da_mb_connection.append(CustomLinear(weight_inh_test * con_matrix7, con_matrix7))
    # da-apl
    con_matrix8 = torch.ones((num_da, num_apl), dtype=torch.float)
    da_mb_connection.append(CustomLinear(weight_inh_test * con_matrix8, con_matrix8))
    # apl-da
    con_matrix9 = torch.ones((num_apl, num_da), dtype=torch.float)
    da_mb_connection.append(CustomLinear(weight_inh_test * con_matrix9, con_matrix9))
    # 1-da
    con_matrix10 = torch.ones((num_da), dtype=torch.float)
    da_mb_connection.append(CustomLinear(weight_exc * con_matrix10, con_matrix10))
    # da-mbon
    con_matrix11 = torch.ones((num_da,num_action), dtype=torch.float)
    da_mb_connection.append(CustomLinear(weight_exc * con_matrix11, con_matrix11))

    #0 input-vis 1 vis-kc 2 kc-mbon 3-mbon-mbon 4 kc-apl 5 apl-kc 6 da-apl 7 apl-da 8 input-da
    t1 = torch.zeros((test_num), dtype=torch.float)
    t2 = torch.zeros((test_num), dtype=torch.float)
    for c in range(t1.shape[0]):
        DA_MB_test = droDMTestNet(copy.deepcopy(da_mb_connection))
        DA_MB_test.reset()
        Gt = torch.tensor([0, (c * 0.1), 0, 0, 0.5])
        BT = torch.tensor([0, 0, (c * 0.1), 0.5, 0])
        input = Gt - BT  # input Gt
        input[input < 0] = 0
        count = torch.zeros((num_action), dtype=torch.float)
        for i_train in range(500):
            if i_train<10 and i_train%2==0:
                input_da = torch.tensor([0.5])
            else:
                input_da = torch.tensor([0.0])
            GT_out, dwkc, dwapl= DA_MB_test(input,input_da)
            DA_MB_test.UpdateWeight(5, dwkc)
            DA_MB_test.UpdateWeight(4, dwapl)
            count += GT_out
        t1[c] = count[0]
        t2[c] = count[1]
    p2 = (t1 - t2) / (t1 + t2)
    print(t1, t2, p2)
    MB_test = MB.getweight()
    for i in range(len(train_weight)):
        print("weight after learning:", train_weight[i].weight.data)


x = torch.arange(0, test_num)
x=x*0.1
plt.figure()
A,=plt.plot(x, p1,label="linear")
B,=plt.plot(x, p2,label="non-linear")
font1 = {'family' : 'Times New Roman','weight' : 'normal','size' : 15,}
plt.xlabel("color intensity",font1)
plt.ylabel("PI",font1)
plt.legend(handles=[A,B],prop=font1)
plt.show()