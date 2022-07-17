import numpy as np
import torch,os,sys
from torch import nn
from torch.nn import Parameter 

import abc
import math
from abc import ABC

import torch.nn.functional as F
import matplotlib.pyplot as plt
#from BrainCog.base.strategy.surrogate import *
from BrainCog.base.node.node import IFNode
from BrainCog.base.learningrule.STDP import STDP,MutliInputSTDP
from BrainCog.base.connection.CustomLinear import CustomLinear
from BrainCog.base.brainarea.basalganglia import basalganglia
from BrainCog.model_zoo.bdmsnn import BDMSNN

from robomaster import robot
import time

def chooseAct(Net,input,weight_trace_d1,weight_trace_d2):
    """
    根据输入选择行为
    :param Net: 输入BDM-SNN网络
    :param input: 输入电流 编码状态的脉冲
    :param weight_trace_d1: 不断累积保存资格迹
    :param weight_trace_d2: 不断累积保存资格迹
    :return: 返回选择的行为、资格迹和网络
    """
    for i_train in range(500):
        out, dw = Net(input)
        # rstdp
        weight_trace_d1 *= trace_decay
        weight_trace_d1 += dw[0][0]
        weight_trace_d2 *= trace_decay
        weight_trace_d2 += dw[1][0]
        if torch.max(out) > 0:
            return torch.argmax(out),weight_trace_d1,weight_trace_d2,Net

def updateNet(Net,reward, action, state,weight_trace_d1,weight_trace_d2):
    """
    更新网络
    :param Net: BDM-SNN网络
    :param reward: 获得的奖励
    :param action: 执行的行为
    :param state: 执行行为前的状态
    :param weight_trace_d1: 直接通路累积的资格迹
    :param weight_trace_d2: 间接通路累积的资格迹
    :return: 更新后的网络
    """
    r = torch.ones((num_state, num_state * num_action), dtype=torch.float)
    r[state, state * num_action + action] = reward
    dw_d1 = r * weight_trace_d1
    dw_d2 = -1 * r * weight_trace_d2
    Net.UpdateWeight(0, state,num_action,dw_d1)
    Net.UpdateWeight(1, state,num_action,dw_d2)
    return Net

if __name__=="__main__":
    """
    定义无人机 大疆Tello Talent 
    定义BDM-SNN网络
    用户自定义状态空间、奖励函数，调用行为选择及网络更新
    """
    #define UAV
    tl_drone = robot.Drone()
    tl_drone.initialize()
    tl_flight = tl_drone.flight
    tl_flight.takeoff().wait_for_completed()

    #define Net
    num_state=9
    num_action=2
    weight_exc=1
    weight_inh=-0.5
    trace_decay = 0.8
    DM=BDMSNN(num_state,num_action,weight_exc,weight_inh,"lif")
    con_matrix1 = torch.zeros((num_state, num_state * num_action), dtype=torch.float)
    for i in range(num_state):
        for j in range(num_action):
            con_matrix1[i, i * num_action + j] = weight_exc
    weight_trace_d1 = torch.zeros(con_matrix1.shape, dtype=torch.float)
    weight_trace_d2 = torch.zeros(con_matrix1.shape, dtype=torch.float)
    iteration=0
    while iteration < 200:
        input = torch.zeros((num_state), dtype=torch.float)
        #users define the judgestate function
        state=1
        input[state]=2
        action,weight_trace_d1,weight_trace_d2,DM = chooseAct(DM,input,weight_trace_d1,weight_trace_d2)
        #uav do action
        if action==0:
            tl_flight.forward(distance=20).wait_for_completed()
        if action == 1:
            # flying left
            tl_flight.rc(a=20, b=0, c=0, d=0)
            time.sleep(4)
        if action == 2:
            # flying right
            tl_flight.rc(a=-20, b=0, c=0, d=0)
            time.sleep(3)
        if action == 3:
            tl_flight.backward(distance=20).wait_for_completed()
       #users define the reward function
        reward =1

        DM=updateNet(DM,reward, action, state,weight_trace_d1,weight_trace_d2)
        weight_trace_d1 = torch.zeros(con_matrix1.shape, dtype=torch.float)
        weight_trace_d2 = torch.zeros(con_matrix1.shape, dtype=torch.float)
        DM.reset()

        iteration += 1