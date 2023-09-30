import random
import time
import tkinter as tk
import pandas as pd
from tools.ExperimentEnvGlobalNetworkSurvival import ExperimentEnvGlobalNetworkSurvival
from tools.MazeTurnEnvVec import MazeTurnEnvVec
import numpy as np
from matplotlib import pyplot as plt
steps=500
t=[i for i in range(steps)]
class Agent(object):
    '''个体类'''
    MAZE_R = 6  
    MAZE_C = 6 

    def __init__(self, env,alpha=0.1, gamma=0.9):
        '''初始化'''
        self.states = {} 
        self.actions = 3  
    
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros([32,3])

    def choose_action(self,state,epsilon=0.8):
        '''选择相应的动作。根据当前状态，随机或贪婪，按照参数epsilon'''

        if random.uniform(0, 1) > epsilon:  
            action = random.choice([0,1,2])
        else:
            max_index=(self.q_table[state] == self.q_table[state].max()).nonzero()
            if len(max_index)==1:
                max_qvalue_actions=max_index[0]
            else:
                max_qvalue_actions=max_index[:][1]
            action = random.choice(np.array(max_qvalue_actions))
        return np.array([action])


    def update_q_value(self, state, action, next_state_reward, next_state_q_values):
        self.q_table[state, action] += self.alpha * (
                next_state_reward + self.gamma * next_state_q_values.max() - self.q_table[state, action])

    def add_state(self,X_next):
        x_str = ','.join(str(i) for i in X_next.astype(int))
        if (x_str in self.states) == False:
            self.states[x_str] = max(self.states.values()) + 1
        return self.states[x_str]


    def learn(self, env, episode=100, epsilon=0.8):
        '''q-learning算法'''
        env.reset()
        X=np.array([0,1,0,0])
        sss = ','.join(str(i) for i in X.astype(int))
        self.states[sss] = 0
        for i in range(episode):
            steps=0
            current_state = np.array([0])
            env.env.current_cell=np.array([0])
            X_next, envreward, fitness, infos=env.step(current_state)
            self.add_state(X_next)
            next_state_reward=0
            while next_state_reward==0 and steps<1000:
                current_action = self.choose_action(current_state, epsilon) 
                X_next, next_state_reward, fitness, infos = env.step(current_action)
                next_state_number=self.add_state(X_next)
                next_state_q_values = self.q_table[next_state_number]
                self.update_q_value(current_state, current_action, next_state_reward, next_state_q_values)
                current_state = next_state_number
                steps+=1

    def play(self, env):
        step=0
        self.learn(env, epsilon=0.8)
        current_state = np.array([0])
        env.env.current_cell = np.array([0])
        X_next, envreward, fitness, infos = env.step(current_state)
        self.add_state(X_next)
        env_r=[]
        rsum=0
        old_dis=13

        while step<steps:
            current_action = self.choose_action(current_state, 1)
            X_next, envreward, fitness, infos = env.step(current_action)
            envreward=envreward[0]
            food_pos = env.env.food_pos[:, 0, :2]
            agent_pos = env.env.agents_pos
            dis = ((agent_pos - food_pos) ** 2).sum(1)
            reward =np.array((np.sqrt(old_dis)-np.sqrt(dis))>0,dtype=int)[0]
            if reward==0:
                reward=-1
            elif reward==1:
                reward=1
            if envreward==1:
                reward=3
            elif envreward==-1:
                reward=-3
            next_state_number = self.add_state(X_next)
            rsum+=reward
            current_state = next_state_number
            env_r.append(rsum)
            step+=1
        return np.array(env_r)


def QQ():
    steps=500
    env = MazeTurnEnvVec(1, n_steps=steps)
    data_env = ExperimentEnvGlobalNetworkSurvival(env)
    agent = Agent(data_env)  
    r=agent.play(data_env)
    return r

np.save('./ql.npy',QQ())
