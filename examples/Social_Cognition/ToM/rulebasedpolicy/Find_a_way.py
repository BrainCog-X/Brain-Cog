# main.py

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

from rulebasedpolicy.random_map import *
from rulebasedpolicy.a_star import *
from env.env3_train_env01 import FalseBelief_env1


def Find_a_way(size, board, start_x, start_y, end_x, end_y):
    map = RandomMap(size=size, board=board)

    for i in range(map.size):
        for j in range(map.size):
            if map.IsObstacle(i,j):
                rec = Rectangle((i, j), width=1, height=1, color='gray')
            else:
                rec = Rectangle((i, j), width=1, height=1, edgecolor='gray', facecolor='w')

    rec = Rectangle((start_x, start_y), width = 1, height = 1, facecolor='b')

    rec = Rectangle((end_x, end_y), width = 1, height = 1, facecolor='r')

    A_star = AStar(map)
    action_seq = A_star.RunAndSaveImage( start_x, start_y, end_x, end_y)#ax, plt,
    return action_seq

