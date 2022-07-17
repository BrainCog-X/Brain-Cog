import numpy as np
from rulebasedpolicy.point import *


class RandomMap:
    def __init__(self, size, board):
        self.size = size
        self.board = board
        self.obstacle = size//8
        self.GenerateObstacle()

    def GenerateObstacle(self):
        self.obstacle_point = []
        # Generate an obstacle in the middle
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i,j] == 5:
                    self.obstacle_point.append(Point(j, 4-i))



    def IsObstacle(self, i ,j):
        for p in self.obstacle_point:
            if i==p.x and j==p.y:
                return True
        return False