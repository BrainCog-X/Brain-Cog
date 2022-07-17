import sys
import time

import numpy as np

from matplotlib.patches import Rectangle

from rulebasedpolicy.point import *
from rulebasedpolicy.random_map import *

class AStar:
    def __init__(self, map):
        self.map=map
        self.open_set = []
        self.close_set = []

    def BaseCost(self, p):
        x_dis = p.x
        y_dis = p.y
        # Distance to start point
        return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)

    def HeuristicCost(self, p):
        x_dis = self.map.size - 1 - p.x
        y_dis = self.map.size - 1 - p.y
        # Distance to end point
        return x_dis + y_dis + (np.sqrt(2) - 2) * min(x_dis, y_dis)

    def TotalCost(self, p):
        return self.BaseCost(p) + self.HeuristicCost(p)

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0:
            return False
        if x >= self.map.size or y >= self.map.size:
            return False
        return not self.map.IsObstacle(x, y)

    def IsInPointList(self, p, point_list):
        for point in point_list:
            if point.x == p.x and point.y == p.y:
                return True
        return False

    def IsInOpenList(self, p):
        return self.IsInPointList(p, self.open_set)

    def IsInCloseList(self, p):
        return self.IsInPointList(p, self.close_set)

    def IsStartPoint(self, p, start_x, start_y):
        return p.x == start_x and p.y ==start_y

    def IsEndPoint(self, p, end_x, end_y):
        return p.x == end_x and p.y == end_y###############

    def SaveImage(self, plt):
        millis = int(round(time.time() * 1000))
        filename = './' + str(millis) + '.png'
        plt.savefig(filename)

    def ProcessPoint(self, x, y, parent):
        if not self.IsValidPoint(x, y):
            return # Do nothing for invalid point
        p = Point(x, y)
        if self.IsInCloseList(p):
            return # Do nothing for visited point
        # print('Process Point [', p.x, ',', p.y, ']', ', cost: ', p.cost)
        if not self.IsInOpenList(p):
            p.parent = parent
            p.cost = self.TotalCost(p)
            self.open_set.append(p)

    def SelectPointInOpenList(self):
        index = 0
        selected_index = -1
        min_cost = sys.maxsize
        for p in self.open_set:
            cost = self.TotalCost(p)
            if cost < min_cost:
                min_cost = cost
                selected_index = index
            index += 1
        return selected_index

    def BuildPath(self, p,  start_time, start_x, start_y, end_x, end_y):#ax, plt,
        path = []
        record = []
        while True:
            path.insert(0, p) # Insert first
            if self.IsStartPoint(p, start_x, start_y):
                break
            else:
                p = p.parent
        p_x=start_x
        p_y=start_y
        for p in path:
            if abs(p.x-p_x) == abs(p.y-p_y) == 1:
                # rec = Rectangle((p_x, p.y), 1, 1, color='g')
                # rec = Rectangle((p.x, p.y), 1, 1, color='g')
                # ax.add_patch(rec)
                # plt.draw()
                # self.SaveImage(plt)
                if abs(end_x - start_x) >= abs(end_y - start_y):
                    record.append((p.x, p_y))
                    record.append((p.x, p.y))
                else:
                    record.append((p_x, p.y))
                    record.append((p.x, p.y))
            else:
                rec = Rectangle((p.x, p.y), 1, 1, color='g')
                # ax.add_patch(rec)
                # plt.draw()
                # self.SaveImage(plt)
                record.append((p.x, p.y))
            p_x = p.x
            p_y = p.y

        end_time = time.time()
        # print('===== Algorithm finish in', int(end_time-start_time), ' seconds')
        return record

    def RunAndSaveImage(self,  start_x, start_y, end_x, end_y):#ax, plt,
        start_time = time.time()

        start_point = Point(start_x, start_y)############################
        start_point.cost = 0
        self.open_set.append(start_point)

        while True:
            index = self.SelectPointInOpenList()
            if index < 0:
                print('No path found, algorithm failed!!!')
                # self.SaveImage(plt)
                return
            p = self.open_set[index]
            # rec = Rectangle((p.x, p.y), 1, 1, color='c')
            # ax.add_patch(rec)
            # self.SaveImage(plt)

            if self.IsEndPoint(p, end_x, end_y):
                return self.BuildPath(p,  start_time, start_x, start_y, end_x, end_y)#ax, plt,

            del self.open_set[index]
            self.close_set.append(p)

            # Process all neighbors
            x = p.x
            y = p.y
            self.ProcessPoint(x-1, y+1, p)
            self.ProcessPoint(x-1, y, p)
            self.ProcessPoint(x-1, y-1, p)
            self.ProcessPoint(x, y-1, p)
            self.ProcessPoint(x+1, y-1, p)
            self.ProcessPoint(x+1, y, p)
            self.ProcessPoint(x+1, y+1, p)
            self.ProcessPoint(x, y+1, p)




