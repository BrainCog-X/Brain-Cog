import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rulebasedpolicy.load_statedata import *
import math
np.set_printoptions(threshold = np.inf)

def data():
    batch_size = 45
    # read data
    txt = os.path.join(sys.path[0],'rulebasedpolicy', 'train.txt')
    train_loader=get_dataloader(mode=txt, num=batch_size ,batch=batch_size)

    for data in train_loader:
        A = data["A"].numpy()
        B = data["B"].numpy()
        B = B.reshape(batch_size, -1, 5, 5)
        A = A.reshape(batch_size, 5, 5)
    # distant between A and Wall(B)计算智能体与墙之间的距离
    A_train = np.sum(np.square(np.argwhere(A==8)-np.argwhere(A==5)), axis = 1)
    dist_AW = 1               #指定一个特定的距离1,2,4,5,9,10 distant between agent and wall############
    o_idx = np.argwhere(A_train == dist_AW) #找到固定距离对应的所有矩阵Find all matrices corresponding to a fixed distance
    return B, A_train


def flip180(arr):
    """
    翻转180度
    @param arr:
    @return:
    """
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def flip90_left(arr):
    """
    向左翻转90度逆时针
    @param arr:
    @return:
    """
    new_arr = np.transpose(arr)
    new_arr = new_arr[::-1]
    return new_arr

def flip90_right(arr):
    """
    向右翻转90度顺时针
    @param arr:
    @return:
    """
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    new_arr = np.transpose(new_arr)[::-1]
    return new_arr

def gain_env(obs, agent, wall):
    """
    Aim:可以根据多个部分观察在一起拼成一个大的环境
    @param obs:多组观测
    @param agent:代表智能体的参数，这里默认用8来表示
    @param wall:代表墙的参数，这里默认用5来表示
    @return:拼成的环境
    obs_random:随便选择一个矩阵作为based环境
    obs[i]:其他用来补全的矩阵
    x_a, y_a:based环境-智能体坐标
    x_w, y_w:based环境-wall坐标
    x_t, y_t:其他环境-智能体坐标
    x_tt, y_tt:其他环境-wall坐标
    """

    obs_random = obs[0]
    for i in range(1, obs.shape[0]):
        #based env
        x_a, y_a = np.argwhere(obs_random == agent)[0]
        x_w, y_w = np.argwhere(obs_random == wall)[0]
        #external env
        x_t, y_t = np.argwhere(obs[i] == agent)[0]
        x_tt, y_tt = np.argwhere(obs[i] == wall)[0]

        h, l = obs_random.shape
        delta_up = max(x_t - x_a,0)
        delta_down = max(5-x_tt - (h-x_w),0)
        delta_left = max(y_t - y_a,0)
        delta_right = max(5-y_tt - (l-y_w),0)

        obs_random = np.r_[np.ones((delta_up, l)), obs_random] if delta_up != 0 else obs_random
        h, l = obs_random.shape
        obs_random = np.r_[obs_random, np.ones((delta_down, l))] if delta_down != 0 else obs_random
        h, l = obs_random.shape
        obs_random = np.c_[np.ones((h, delta_left)), obs_random] if delta_left != 0 else obs_random
        h, l = obs_random.shape
        obs_random = np.c_[obs_random, np.ones((h, delta_right))] if delta_right != 0 else obs_random
        obs_random = obs_random.astype(np.int)
        #based env
        x_a, y_a = np.argwhere(obs_random == agent)[0]
        up = x_a - x_t
        left = y_a - y_t
        obs_random[up:up+5, left:left+5] = obs_random[up:up+5, left:left+5] & obs[i]

    return obs_random

def shelter_env(obs):
    """
    Aim:用gain_env环境中的图，来描述更复杂的环境的遮挡关系
    @param obs:复杂的环境
    @return:环境的遮挡关系
    """
    # print(obs,'----------------')
    position_A = np.argwhere(obs==8)
    position_W = np.argwhere(obs==5)
    # print(position_W)
    position = np.sum(np.square(np.argwhere(obs == 8) - np.argwhere(obs == 5)), axis=1) #numpy  (walls,)
    # print(position_W, position_A,position)
    shelter_env_i = np.ones((5,5)).astype(np.int)
    B, A_train = data()
    for i in range(position.size):
        o_idx = np.argwhere(A_train == position[i])
        if o_idx.size == 0:
            break
        else:
            model = gain_env(B[o_idx].reshape(-1, 5, 5), 8 ,5).astype(np.int)

            # print(model,'=============')

            if (position_A[0,0] > position_W[i,0] and position_A[0,1] < position_W[i,1] and \
                    position_A[0, 0] - position_W[i, 0] < -position_A[0, 1] + position_W[i, 1])\
                or\
                (position_A[0, 0] < position_W[i, 0] and position_A[0, 1] < position_W[i, 1] and \
                 -position_A[0, 0] + position_W[i, 0] > -position_A[0, 1] + position_W[i, 1])\
                or\
                (position_A[0, 0] > position_W[i, 0] and position_A[0, 1] > position_W[i, 1] and \
                 position_A[0, 0] - position_W[i, 0] > position_A[0, 1] - position_W[i, 1])\
                or\
                (position_A[0, 0] < position_W[i, 0] and position_A[0, 1] > position_W[i, 1] and \
                 -position_A[0, 0] + position_W[i, 0] < position_A[0, 1] - position_W[i, 1]):

                model = np.flip(model, 0)
                # print(model, '-=-=-=-=-=-=-====')
                model = flip90_right(model)
                # print(model,'-=-=-=-=-=-=-====')
            if position_A[0, 0] >= position_W[i, 0] and position_A[0, 1] > position_W[i, 1]:
                model = flip180(model)
            elif position_A[0, 0] > position_W[i, 0] and position_A[0, 1] <= position_W[i, 1]:
                model = flip90_left(model)
            elif position_A[0, 0] < position_W[i, 0] and position_A[0, 1] >= position_W[i, 1]:
                model = flip90_right(model)
            else:
                model = model

            x_t, y_t = np.argwhere(model == 8)[0]
            if y_t<position_A[0, 1]:
                model = np.c_[np.ones((model.shape[0], position_A[0, 1]-y_t)).astype(np.int), model]

            if x_t < position_A[0, 0]:
                model = np.r_[np.ones((position_A[0, 0] - x_t, model.shape[1])).astype(np.int), model]

            if model.shape[0] - x_t < 5 - position_A[0, 0]:
                model = np.r_[model, np.ones((5 - position_A[0, 0] - model.shape[0] + x_t, \
                                              model.shape[1])).astype(np.int)]

            if model.shape[1] - y_t < 5 - position_A[0, 1]:
                model = np.c_[model, np.ones((model.shape[0], \
                                              5 - position_A[0, 1]-model.shape[1] + y_t)).astype(np.int)]
            model = model.astype(np.int)
            # print(model,'...........')
            x_t, y_t = np.argwhere(model == 8)[0]
            shelter_env_i = model[x_t-position_A[0, 0]:x_t-position_A[0, 0]+5, \
                           y_t-position_A[0, 1]:y_t-position_A[0, 1]+5] & shelter_env_i
            # print(shelter_env_i,'66666666666666')

    shelter_env_i[np.argwhere(obs==5)[:,0], np.argwhere(obs==5)[:,1]] = 5
    shelter_env_i[position_A[0,0], position_A[0, 1]] = 8
    shelter_env_i = shelter_env_i.astype(np.int)

    return shelter_env_i

def big_env(env):
    h, l = np.shape(env)
    res = np.ones((h+2, l+2))*5
    res[1:6, 1:6] = env
    return res.astype(np.int)

# model = gain_env(B[o_idx].reshape(-1, 5, 5), 8 ,5)

# obs_esti = shelter_env(np.array([[1,1,1,5,5],[1,1,1,5,5],[1,1,1,1,5],[1,8,1,1,1],[1,1,1,1,1]]))
# env = big_env(obs_esti)
# print(env)

