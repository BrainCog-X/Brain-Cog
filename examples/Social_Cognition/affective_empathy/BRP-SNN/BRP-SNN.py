import os
import sys
# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import imageio
from env_poly_SNN import Maze
from env_two_poly_SNN import Maze2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
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



X=np.array([[0],
            [1],
            [2],
            [3]])
Y=np.array([[0,-40],
            [0,40],
            [40,0],
            [-40,0]
            ])

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


class BNESNN(BrainArea):
    """
    负面情绪网络
    """
    def __init__(self,):
        super().__init__()
        self.node = [IFNode() for i in range(5)]
        self.connection = []
        
        con_matrix0 = torch.eye(12, 12)*6
        self.connection.append(CustomLinear(con_matrix0))#input-state
        
        con_matrix1 = torch.zeros((12, 24), dtype=torch.float)   
        self.connection.append(CustomLinear(con_matrix1))#state-prediction
        
        con_matrix2 = torch.eye(24, 24)*6
        self.connection.append(CustomLinear(con_matrix2))#input-prediction
        
        con_matrix3 = torch.eye(24, 24)*6
        self.connection.append(CustomLinear(con_matrix3))#input-sensory
        
        con_matrix4 = torch.eye(24, 24)*6
        self.connection.append(CustomLinear(con_matrix4))#sensory-error
        
        con_matrix5 = torch.eye(24, 24)*(-6)
        self.connection.append(CustomLinear(con_matrix5))#prediction-error
        
        con_matrix6 = torch.zeros((24, 24), dtype=torch.float)   
        p=0.5        
        if p==0.25:
            con_matrix6[:,0:3]=1
        if p==0.5:
            con_matrix6[:,0:6]=1
        if p==0.75:
            con_matrix6[:,0:9]=1
        if p==1:
            con_matrix6[:,0:12]=1
        self.connection.append(CustomLinear(con_matrix6))#error-pain
        
        self.stdp = []
        self.stdp.append(STDP(self.node[0], self.connection[0]))#node0-state,stdp0
        self.stdp.append(MutliInputSTDP(self.node[1], [self.connection[1], self.connection[2]]))#node1-prediction,stdp1
        self.stdp.append(STDP(self.node[3], self.connection[3]))#node3-sensory,stdp2
        self.stdp.append(MutliInputSTDP(self.node[2], [self.connection[4], self.connection[5]]))#node2-error,stdp3
        self.stdp.append(STDP(self.node[1], self.connection[1]))#node1-prediction,stdp4
        self.stdp.append(STDP(self.node[4], self.connection[6]))#node4-pain,stdp5
    def forward(self, x1,x2):
        """
        计算前向传播过程,训练过程
        """
        out__s, dw0 = self.stdp[0](x1)#node0
        out__p,dw = self.stdp[1](out__s,x2)#node1
    
        return dw,out__s,out__p
    
    def calculate_error(self, x1,x2):
        """
        测试过程
        """
        out__s,dw = self.stdp[0](x1)#node0-state,stdp0
        out__pre,dw= self.stdp[4](out__s)#node1-prediction,stdp1
        out__sensory,dw = self.stdp[2](x2)#node3-sensory,stdp2
        out__error,dw = self.stdp[3](out__sensory,out__pre)#node2-error,stdp3
        out__pain,dw = self.stdp[5](out__error)#node4-pain,stdp5
        
    
        return out__s,out__pre,out__sensory,out__error,out__pain
        
    def UpdateWeight(self, i, dw, delta):
        """
        更新第i组连接的权重 根据传入的dw值
        :param i: 要更新的连接的索引
        :param dw: 更新的量
        :return: None
        """
        self.connection[i].update(dw*delta)
        self.connection[i].weight.data= torch.clamp(self.connection[i].weight.data,0,6)
        
    def reset(self):
        """
        reset神经元或学习法则的中间量
        :return: None
        """
        for i in range(5):
            self.node[i].n_reset()
        for i in range(len(self.stdp)):
            self.stdp[i].reset()

         
def GRF(X,N):
    gauss_neuron = 12  
    center = np.ones((gauss_neuron, 1))
    width = 1 / 15

    for i in range(len(center)):  
        center[i] = (2 * i - 3) / 20  
    x = np.arange(0, 1, 0.0001)  

    num_features = N

    gauss_recpt_field = np.zeros((gauss_neuron, len(x)))  
    for i in range(gauss_neuron):
        gauss_recpt_field[i, :] = np.exp(-(x - center[i]) ** 2 / (2 * width * width))  

    def gauss_response(inputs,num_features):
        spike_time = np.zeros((gauss_neuron, num_features))
        # input: shape [1, features]
        # output: shape [gaussian neurons*features] spiking time
        for i in range(num_features):
            for j in range(gauss_neuron):
                spike_time[j, i] = gauss_recpt_field[j, inputs[i]]  #entry gauss function
        spikes = []
        for i in range(spike_time.shape[1]):
            spikes.extend(spike_time[:, i])
        return np.array(spikes)


    gauss_neurons = gauss_neuron * N
   
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = (X * 10000).astype(int)  #10000
    X[X == 10000] = 9999 
    input_spike = np.zeros((X.shape[0], gauss_neurons))  
    for i in range(X.shape[0]):
        input_spike[i, :] = gauss_response(X[i, :],num_features)
    input_spike[input_spike < 0.1] = 0  
    input_spike = np.around(100 * (1 - input_spike))  
    input_spike[input_spike == 0] = 1
    input_spike[input_spike == 100] = 0
    state=[]
    for i in range(len(X)):
        aa=[]
        for j in range(gauss_neurons):
            if input_spike[i][j] != 0:
                number=input_spike[i][j]
                aa.append((int(number),j))
        state.append(aa)
        
    return state
             
                
def encode(input,n_neuron):
    a=len(input)
    input_encode = []
    for i in range(n_neuron):
        temp = np.zeros([100, ])
        input_encode.append(temp)
    for j in range(a):
        s=input[j][0]
        n=input[j][1]
        input_encode[n][s]=1

    return input_encode


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
        
        con_matrix0 = torch.eye(24, 24)*6
        self.connection.append(CustomLinear(con_matrix0))#input-emotion
        
        con_matrix1 = torch.zeros((24, 50), dtype=torch.float)
        for j in range(50):
            if j in np.arange(0,25,1):
                for i in np.arange(0, 12, 1):
                    con_matrix1[i,j] =2
            if j in np.arange(25,50,1):
                for i in np.arange(12, 24, 1):
                    con_matrix1[i,j] =2    
        self.connection.append(CustomLinear(con_matrix1))#emotion-ifg
        
        con_matrix2 = torch.zeros((24, 50), dtype=torch.float)  
        self.connection.append(CustomLinear(con_matrix2))#perception-ifg
        
        con_matrix3 = torch.eye(24, 24)*6
        self.connection.append(CustomLinear(con_matrix3))#input-perception
        
        con_matrix4=torch.zeros((24,10), dtype=torch.float)
        for j in range(10):
            if j in np.arange(0,5,1):
                for i in np.arange(0, 12, 1):
                    con_matrix4[i,j] =2
            if j in np.arange(5,10,1):
                for i in np.arange(12, 24, 1):
                    con_matrix4[i,j] =2
        self.connection.append(CustomLinear(con_matrix4))#emotion-sma
        
        con_matrix5=torch.zeros((24,10), dtype=torch.float)
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

def BNESNN_train():  
    
    state=GRF(X,1)
    prediction=GRF(Y,2)

    T=100 
    epoch=10
    for k in range(epoch):
        print('epoch:',k)
        for n in range(4):
            snn1.reset()
            train_state = np.array(encode(state[n], 12))
            train_state=torch.tensor(train_state,dtype=torch.float32)
            train_prediction = np.array(encode(prediction[n], 24))
            train_prediction=torch.tensor(train_prediction,dtype=torch.float32)
            for i in range(T):
                OUTPUT = snn1(train_state[:,i],train_prediction[:,i])
                snn1.UpdateWeight(1,OUTPUT[0][0],1)



def BAESNN_train():  
    s = env.reset()
    env._set_danger()
    env._set_wall()
    pain=0
    i=0
    set_pain=0
    env._set_switch()
    for i in range(100):
        snn1.reset()
        T=100
        pain=0
        print('**************step:',i)
        env.render()
        
        action = np.random.choice(list(range(env.n_actions)))
        print('action:',action)
        d,d_pre,s_,sss = env.step(s, action, pain)
        print('d:',d,'d_pre:',d_pre,'sss:',sss)
        env.render()
            
        while (d==np.array([0,0])).all():
            action = np.random.choice(list(range(env.n_actions)))
            print('action:',action)
            d,d_pre,s_,sss = env.step(s, action, pain)
            print('d:',d,'d_pre:',d_pre,'sss:',sss)
            env.render()
        
        
        
        aa=np.argwhere(X==action)[0][0]
        for i in range(4):
            if (Y[i]==d).all():
                b=i
        print('aa:',aa,'b:',b)       
        state=GRF(X,1)
        prediction=GRF(Y,2)
        x=encode(state[aa],12)
        y=encode(prediction[b],24)
        train_state = np.array(x)
        train_state=torch.tensor(train_state,dtype=torch.float32)
        train_prediction = np.array(y)
        train_prediction=torch.tensor(train_prediction,dtype=torch.float32)
        OUT_PAIN=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0.]])  
        spike_pain=[]
        spike_error=[]
        for i in range(T):
            OUTPUT_TEST = snn1.calculate_error(train_state[:,i],train_prediction[:,i])
            spike_pain.append(OUTPUT_TEST[4])
            spike_error.append(OUTPUT_TEST[3])
            if OUTPUT_TEST[3].sum() != 0:
                print('OUTPUT_TEST3:',i,OUTPUT_TEST[3])
            if OUTPUT_TEST[4].sum() != 0:#pain brain area
                print('OUTPUT_TEST4:',i,OUTPUT_TEST[4])
                OUT_PAIN=OUTPUT_TEST[4]
                pain=1
                set_pain=1
        spike_pain = torch.stack(spike_pain)
        spike_error=torch.stack(spike_error)
        if pain==1:
            spike_rate_vis_1d(spike_error)
            spike_rate_vis_1d(spike_pain)
        print('pain:',pain)
        
        
        
        
        
        snn2.reset()
        T2=20
        X1= OUT_PAIN.view(1, -1) 
        X2=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0.]])  
        print('X1,X2:',X1,X2)
        for i in range(T2):
            if i>=2:
                X2=X1
            OUTPUT = snn2(X1,X2)
            snn2.UpdateWeight(2,OUTPUT[0][1],0.01)
            snn2.UpdateWeight(5,OUTPUT[1][1],-0.1)
            if OUTPUT[2][0][0]==1:
                env.canvas.itemconfig(env.rect, fill="red", outline='red')
            if OUTPUT[2][0][0]==0:
                env.canvas.itemconfig(env.rect, fill="green", outline='green')
        env.render()
        
        print('out_ifg:',OUTPUT[2])
        print('out_sma:',OUTPUT[3])
        print('out_m1:',OUTPUT[4])
        print('con2:',snn2.connection[2].weight.data)
        print('con5:',snn2.connection[5].weight.data)
        
        s = s_
        if set_pain==1 and pain==0:
            env.render()
            break
    env.destroy()
                

def BAESNN_test():
    s1,s=env2.reset()
    pain=0
    pain1 = 0
    i=0
    set_pain=0
    
    for i in range(100):
        
        snn1.reset()
        T=100
        pain=0
        print('**************test_step:',i)
        env2.render()
        
        action1 = np.random.choice(list(range(env.n_actions)))
        print('action1:',action1)
        d,d_pre,s1_,sss = env2.step(s1, action1, pain1)
        print('d:',d,'d_pre:',d_pre,'sss:',sss)
        env2.render()
            
        while (d==np.array([0,0])).all():
            action1 = np.random.choice(list(range(env.n_actions)))
            print('action1:',action1)
            d,d_pre,s1_,sss = env2.step(s, action1, pain1)
            print('d:',d,'d_pre:',d_pre,'sss:',sss)
            env2.render()
        
        
        
        aa=np.argwhere(X==action1)[0][0]
        for i in range(4):
            if (Y[i]==d).all():
                b=i
        # print('aa:',aa,'b:',b)       
        state=GRF(X,1)
        prediction=GRF(Y,2)
        x=encode(state[aa],12)
        y=encode(prediction[b],24)
        train_state = np.array(x)
        train_state=torch.tensor(train_state,dtype=torch.float32)
        train_prediction = np.array(y)
        train_prediction=torch.tensor(train_prediction,dtype=torch.float32)
        OUT_PAIN=torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0.]])  
        for i in range(T):
            OUTPUT_TEST = snn1.calculate_error(train_state[:,i],train_prediction[:,i])
            if OUTPUT_TEST[3].sum() != 0:
                print('OUTPUT_TEST3:',i,OUTPUT_TEST[3])
            if OUTPUT_TEST[4].sum() != 0:#pain brain area
                print('OUTPUT_TEST4:',i,OUTPUT_TEST[4])
                OUT_PAIN=OUTPUT_TEST[4]
                pain1=1
                set_pain=1
        print('pain1:',pain1)
        

        
        env2.generate_expression1(pain1)
        
        
        snn2.reset()
        T2=20
        X3= OUT_PAIN.view(1, -1) 
       
        for i in range(T2):
            OUT=snn2.empathy(X3)
            print('out_ifg:',OUT[0])
        
        if OUT[0][0][0]==1:
            pain=1        
        
        if OUT[0][0][0]==0:
            pain=0  
                
        if pain==1:
            env2.agent_help()
            
        s1 = s1_
        env2.render()

        if pain==0 and set_pain==1:
            env2.render()
            break
  
    env2.destroy()
  





if __name__ == "__main__":
    env = Maze() 
    snn1 = BNESNN()
    snn2 = BAESNN() 
    BNESNN_train()
    BAESNN_train()
    env.mainloop()
    
    env2 = Maze2()
    BAESNN_test()
    env2.mainloop()