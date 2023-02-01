import imageio
from env_poly import Maze
from env_two_poly import Maze2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


import torch, os, sys
from torch import nn
from torch.nn import Parameter
import abc
import math
from abc import ABC
import torch.nn.functional as F
from braincog.base.node.node import *
from braincog.base.learningrule.STDP import *
from braincog.base.connection.CustomLinear import *


class BrainArea(nn.Module, abc.ABC):
    """
    脑区基类
    """

    @abc.abstractmethod
    def __init__(self):
        """
        """
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        """
        计算前向传播过程
        :return:x是脉冲
        """

        return x

    def reset(self):
        """
        计算前向传播过程
        :return:x是脉冲
        """

        pass


class BAESNN(BrainArea):
    """
    情感共情网络
    """

    def __init__(self,):
        """
        """
        super().__init__()


        self.node = [IFNode() for i in range(5)]
       
        
        self.connection = []
        
        con_matrix0 = torch.eye(40, 40)*6
        self.connection.append(CustomLinear(con_matrix0))#input-emotion
        
        con_matrix1 = torch.zeros((40, 50), dtype=torch.float)
        for j in range(50):
            if j in np.arange(0,20,1):
                for i in np.arange(0, 20, 1):
                    con_matrix1[i,j] =2
            if j in np.arange(30,50,1):
                for i in np.arange(20, 40, 1):
                    con_matrix1[i,j] =2
            if j in np.arange(20,30,1):
                for i in np.arange(0, 40, 1):
                    con_matrix1[i,j] = 2     
        self.connection.append(CustomLinear(con_matrix1))#emotion-ifg
        
        con_matrix2 = torch.zeros((40, 50), dtype=torch.float)  
        self.connection.append(CustomLinear(con_matrix2))#perception-ifg
        
        con_matrix3 = torch.eye(40, 40)*6
        self.connection.append(CustomLinear(con_matrix3))#input-perception
        
        con_matrix4=torch.zeros((40,10), dtype=torch.float)
        for j in range(10):
            if j in np.arange(0,5,1):
                for i in np.arange(0, 20, 1):
                    con_matrix4[i,j] =2
            if j in np.arange(5,10,1):
                for i in np.arange(20, 40, 1):
                    con_matrix4[i,j] =2
        self.connection.append(CustomLinear(con_matrix4))#emotion-sma
        
        con_matrix5=torch.zeros((40,10), dtype=torch.float)
        self.connection.append(CustomLinear(con_matrix5))#perception-m1
        
        con_matrix6 = torch.eye(10, 10)*6
        self.connection.append(CustomLinear(con_matrix6))#sma-m1
        
        self.stdp = []
        self.stdp.append(STDP(self.node[0], self.connection[0]))#0
        self.stdp.append(STDP(self.node[2], self.connection[3]))#1
        self.stdp.append(MutliInputSTDP(self.node[1], [self.connection[1], self.connection[2]]))#2
        self.stdp.append(MutliInputSTDP(self.node[3], [self.connection[4], self.connection[5]]))#3
        self.stdp.append(STDP(self.node[4], self.connection[6]))#4
        self.stdp.append(STDP(self.node[1],self.connection[2]))#5
        self.stdp.append(STDP(self.node[3],self.connection[5]))#6
    def forward(self, x1,x2):
        """
        计算前向传播过程
        :return:x是脉冲
        """
        out__m, dw0 = self.stdp[0](x1)#node0
        out__p, dw3 = self.stdp[1](x2)#node2
        out__ifg,dw_p_i=self.stdp[2](out__m,out__p)#node1
        out__sma,dw_p_s=self.stdp[3](out__m,out__p)#node3
        out__m1,dw1=self.stdp[4](out__sma)#node4
    
        return dw_p_i,dw_p_s,out__ifg,out__sma,out__m1
    
    def empathy(self,x3):
        out_p,dw2=self.stdp[1](x3)#node2
        out_ifg,dw4=self.stdp[5](out_p)#node1
        out_sma,dw5=self.stdp[6](out_p)#node3
        out_m1,dw6=self.stdp[4](out_sma)#node4
        return out_ifg,out_sma,out_m1
        
    def UpdateWeight(self, i, dw, delta):
        """
        更新第i组连接的权重 根据传入的dw值
        :param i: 要更新的连接的索引
        :param dw: 更新的量
        :return: None
        """
        self.connection[i].update(dw*delta)
        self.connection[i].weight.data= torch.clamp(self.connection[i].weight.data,-1,4)
        
    def reset(self):
        """
        reset神经元或学习法则的中间量
        :return: None
        """
        for i in range(5):
            self.node[i].n_reset()
        for i in range(len(self.stdp)):
            self.stdp[i].reset()

def BAESNN_train():  
    s = env.reset()
    env._set_danger()
    env._set_wall()
    pain=0
    i=0
    set_pain=0
    env._set_switch()
    for i in range(100):
        a.reset()
        T=20
        print('step:',i)
        env.render()
        
        action = np.random.choice(list(range(env.n_actions)))
        s_,s_pre,s_color = env.step(s, action, pain)
        env.render()

        if env.open_door == 1:
            env.render()
        
        true_s_1 = np.array(s_)
        predict_s_1=np.array(s_pre)
        error = true_s_1 - predict_s_1
        error = sum([c * c for c in error])
        if error>=3200:
            error=3200
        if error>0:
            pain=1
        if error==0:
            pain=0

        if pain==0:
            X1=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
            X2=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            env.render()
            
        if pain==1:
            set_pain = 1
            X1=torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            X2=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
            env.render()
            
        for i in range(T):
            if i>=2:
                X2=X1
            OUTPUT = a(X1,X2)
            a.UpdateWeight(2,OUTPUT[0][1],0.01)
            a.UpdateWeight(5,OUTPUT[1][1],-0.1)
        if OUTPUT[2][0][0]==1:
            env.canvas.itemconfig(env.rect, fill="red", outline='red')
        if OUTPUT[2][0][40]==1:
            env.canvas.itemconfig(env.rect, fill="green", outline='green')
        env.render()
        
        print('out_ifg:',OUTPUT[2])
        print('out_sma:',OUTPUT[3])
        print('out_m1:',OUTPUT[4])
        # print('con2:',a.connection[2].weight.data)
        # print('con5:',a.connection[5].weight.data)
        
        s = s_
        if set_pain==1 and pain==0:
            env.render()
    env.destroy()
                

def BAESNN_test():
    a.reset()
    s1,s=env2.reset()
    pain=0
    pain1 = 0
    i=0
    set_pain=0
    
    for i in range(1000):
        env2.render()
        
        s_now = env2.canvas.coords(env2.agent1)

        action1 = np.random.choice([0,1,2,3], p=[0.2, 0.3, 0.3, 0.2])
        if env2.open_door==1 and s_now[0] <(9 / 2) * 40:
            action1 = np.random.choice([0,1,2,3], p=[0.5, 0.0, 0.0, 0.5])

        s1_, s1_pre,s1_color = env2.step1(action1,pain)
        print('s1_color:',s1_color)

        if env2.open_door == 1 :
            env2.render()

        true_s1_1 = np.array(s1_)
        predict_s1_1=np.array(s1_pre)
        error1 = true_s1_1 - predict_s1_1
        error1 = sum([c * c for c in error1])
        if error1>=3200:
            error1=3200

        if error1>0:
            pain=1
            set_pain=1
            
        if error1==0:
            pain=0
        
        env2.generate_expression1(pain)
        
        if s1_color=="red":
            X3=torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        if s1_color=="blue":
            X3=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
            
        a.reset()
        for i in range(20):
            OUT=a.empathy(X3)
            print(OUT)
        if pain==1:
            env2.agent_help()
            
        s1 = s1_
        env2.render()

        if pain==0 and set_pain==1:
            env2.render()
            break
  
    # env2.destroy()
  





if __name__ == "__main__":
    env = Maze() 
    a = BAESNN() 
    BAESNN_train()
    env.mainloop()
    
    env2 = Maze2()
    BAESNN_test()
    env2.mainloop()