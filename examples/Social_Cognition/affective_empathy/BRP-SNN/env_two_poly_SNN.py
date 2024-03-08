import numpy as np
np.random.seed(1)
import tkinter as tk
import time
from PIL import ImageGrab


UNIT = 40   # pixels
MAZE_H = 9  # grid height
MAZE_W = 4 # grid width


class Maze2(tk.Tk, object):
    def __init__(self):
        super(Maze2, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_space1 = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_actions1 = len(self.action_space1)
        self.title('two_agent_empathy')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()
        self.danger=0
        self.action_hurt=0
        self.sensory_hurt = 0
        self.action_hurt1 = 0
        self.sensory_hurt1 = 0
        self.open_door=0

    # create environment
    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_W * UNIT,
                           width=MAZE_H * UNIT)

        # create grids
        for c in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create switch
        self.oval_center = np.array([(MAZE_H * UNIT)/2-UNIT+80, ((MAZE_W+4) * UNIT)/2-UNIT/2-80])
        self.oval = self.canvas.create_oval(
            self.oval_center[0] - 15, self.oval_center[1] - 15,
            self.oval_center[0] + 15, self.oval_center[1] + 15,
            fill='yellow')
        self.switch = self.canvas.coords(self.oval)

        self.orgin1 = np.array([20, 20])
        # 下
        self.points1 = [
            # 左上
            self.orgin1[0] - 15,  # 5
            self.orgin1[1] - 15,  # 5
            # 右上
            self.orgin1[0] + 15,  # 35
            self.orgin1[1] - 15,  # 5
            # 右下+
            self.orgin1[0] + 15,  # 35
            self.orgin1[1],  # 20
            # 顶点
            self.orgin1[0],  # 20
            self.orgin1[1] + 15,  # 35
            # 左下+
            self.orgin1[0] - 15,  # 5
            self.orgin1[1],  # 20
        ]
        self.agent1 = self.canvas.create_polygon(self.points1, outline='black',fill="blue")

        self.orgin = np.array([MAZE_H * UNIT - UNIT / 2, 20])
        # 下
        self.points = [
            # 左上
            self.orgin[0] - 15,  # 5
            self.orgin[1] - 15,  # 5
            # 右上
            self.orgin[0] + 15,  # 35
            self.orgin[1] - 15,  # 5
            # 右下+
            self.orgin[0] + 15,  # 35
            self.orgin[1],  # 20
            # 顶点
            self.orgin[0],  # 20
            self.orgin[1] + 15,  # 35
            # 左下+
            self.orgin[0] - 15,  # 5
            self.orgin[1],  # 20
        ]
        self.agent = self.canvas.create_polygon(self.points, fill="green")

        wall_center = []
        self.wall = []
        for i in range(MAZE_W):
            wall_center.append([])
            self.wall.append([])
        for i in range(MAZE_W):
            wall_center[i] = np.array([(MAZE_H * UNIT) / 2, ((i) * UNIT) + UNIT / 2])
            self.wall[i] = self.canvas.create_rectangle(
                wall_center[i][0] - 20, wall_center[i][1] - 20,
                wall_center[i][0] + 20, wall_center[i][1] + 20,
                fill='grey')

        self.hell1_center = np.array([100, 20])
        self.hell1 = self.canvas.create_oval(
            self.hell1_center[0] - 15, self.hell1_center[1] - 15,
            self.hell1_center[0] + 15, self.hell1_center[1] + 15,
            fill='black')
        self.hell2_center = np.array([60, 100])
        self.hell2 = self.canvas.create_oval(
            self.hell2_center[0] - 15, self.hell2_center[1] - 15,
            self.hell2_center[0] + 15, self.hell2_center[1] + 15,
            fill='black')
        # self.canvas.create_bitmap((40 , 40), bitmap='error')

        self.danger = 1

        self.canvas.pack()

    #reset agent location
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.agent1)
        self.canvas.delete(self.agent)
        self.orgin1 = np.array([20, 20])
        # 下
        self.points1 = [
            # 左上
            self.orgin1[0] - 15,  # 5
            self.orgin1[1] - 15,  # 5
            # 右上
            self.orgin1[0] + 15,  # 35
            self.orgin1[1] - 15,  # 5
            # 右下+
            self.orgin1[0] + 15,  # 35
            self.orgin1[1],  # 20
            # 顶点
            self.orgin1[0],  # 20
            self.orgin1[1] + 15,  # 35
            # 左下+
            self.orgin1[0] - 15,  # 5
            self.orgin1[1],  # 20
        ]
        self.agent1 = self.canvas.create_polygon(self.points1, outline='black',fill="blue")

        self.orgin = np.array([MAZE_H * UNIT - UNIT / 2, 20])
        # 下
        self.points = [
            # 左上
            self.orgin[0] - 15,  # 5
            self.orgin[1] - 15,  # 5
            # 右上
            self.orgin[0] + 15,  # 35
            self.orgin[1] - 15,  # 5
            # 右下+
            self.orgin[0] + 15,  # 35
            self.orgin[1],  # 20
            # 顶点
            self.orgin[0],  # 20
            self.orgin[1] + 15,  # 35
            # 左下+
            self.orgin[0] - 15,  # 5
            self.orgin[1],  # 20
        ]
        self.agent = self.canvas.create_polygon(self.points, fill="green")

        return self.canvas.coords(self.agent1),self.canvas.coords(self.agent)



    def step(self, s, action, pain):
        s1 = self.canvas.coords(self.agent1)
        self.centre1 = [(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2]
        if all(self.centre1 == self.hell1_center):
            self.action_hurt1 = 1
        if all(self.centre1 == self.hell2_center):
            self.action_hurt1 = 1
        
        self.oval_center111 = np.array([(MAZE_H * UNIT) / 2, ((MAZE_W + 4) * UNIT) / 2 - UNIT / 2])
        if all(self.centre1 ==self.oval_center111):
            move = np.array([80, 0])
            self.canvas.move(self.agent1, move[0], move[1])
            s1 = self.canvas.coords(self.agent1)
            self.render()
        self.oval_center111 = np.array([(MAZE_H * UNIT) / 2 - UNIT, ((MAZE_W + 4) * UNIT) / 2 - UNIT / 2])
        if all(self.centre1 ==self.oval_center111):
            move = np.array([80, 0])
            self.canvas.move(self.agent1, move[0], move[1])
            s1 = self.canvas.coords(self.agent1)
            self.render()
        self.oval_center111 = np.array([(MAZE_H * UNIT) / 2 - UNIT*2, ((MAZE_W + 4) * UNIT) / 2 - UNIT / 2])
        if all(self.centre1 == self.oval_center111):
            move = np.array([80, 0])
            self.canvas.move(self.agent1, move[0], move[1])
            s1 = self.canvas.coords(self.agent1)
            self.render()
        self.oval_center111 = np.array([(MAZE_H * UNIT) / 2 - UNIT*3, ((MAZE_W + 4) * UNIT) / 2 - UNIT / 2])
        if all(self.centre1 == self.oval_center111):
            move = np.array([80, 0])
            self.canvas.move(self.agent1, move[0], move[1])
            s1 = self.canvas.coords(self.agent1)
            self.render()
        self.oval_center111 = np.array([(MAZE_H * UNIT) / 2 - UNIT*4, ((MAZE_W + 4) * UNIT) / 2 - UNIT / 2])
        if all(self.centre1 == self.oval_center111):
            move = np.array([80, 0])
            self.canvas.move(self.agent1, move[0], move[1])
            s1 = self.canvas.coords(self.agent1)
            self.render()

        



        #显示当前的动作指令是什么
        self.canvas.delete(self.agent1)
        self.centre = [(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2]

        if action==0:
            self.points0 = [
                # 右下
                self.centre[0] + 15,  # 35
                self.centre[1] + 15,  # 35
                # 左下
                self.centre[0] - 15,  # 5
                self.centre[1] + 15,  # 35
                # 左上+
                self.centre[0] - 15,  # 5
                self.centre[1],  # 20
                # 顶点
                self.centre[0],  # 20
                self.centre[1] - 15,  # 5
                # 右上+
                self.centre[0] + 15,  # 35
                self.centre[1],  # 20
            ]
            if pain==0:
                color="blue"
            if pain == 1:
                color = "red"
            self.agent1 = self.canvas.create_polygon(self.points0, fill=color)
        if action==1:
            self.points1 = [
                # 左上
                self.centre[0] - 15,  # 5
                self.centre[1] - 15,  # 5
                # 右上
                self.centre[0] + 15,  # 35
                self.centre[1] - 15,  # 5
                # 右下+
                self.centre[0] + 15,  # 35
                self.centre[1],  # 20
                # 顶点
                self.centre[0],  # 20
                self.centre[1] + 15,  # 35
                # 左下+
                self.centre[0] - 15,  # 5
                self.centre[1],  # 20
            ]
            if pain==0:
                color="blue"
            if pain == 1:
                color = "red"
            self.agent1 = self.canvas.create_polygon(self.points1, fill=color)
        if action==2:
            self.points2 = [
                # 左下
                self.centre[0] - 15,  # 5
                self.centre[1] + 15,  # 35
                # 左上
                self.centre[0] - 15,  # 5
                self.centre[1] - 15,  # 5
                # 右上+
                self.centre[0],  # 20
                self.centre[1] - 15,  # 5
                # 顶点
                self.centre[0] + 15,  # 35
                self.centre[1],  # 20
                # 右下+
                self.centre[0],  # 20
                self.centre[1] + 15,  # 35
            ]
            if pain==0:
                color="blue"
            if pain == 1:
                color = "red"
            self.agent1 = self.canvas.create_polygon(self.points2, fill=color)
        if action==3:
            self.points3 = [
                # 右上
                self.centre[0] + 15,  # 20+15
                self.centre[1] - 15,  # 20-15
                # 右下
                self.centre[0] + 15,  # 20+15
                self.centre[1] + 15,  # 20+15
                # 左下+
                self.centre[0],  # 20
                self.centre[1] + 15,  # 20+15
                # 顶点
                self.centre[0] - 15,  # 20-15
                self.centre[1],  # 20
                # 左上+
                self.centre[0],  # 20
                self.centre[1] - 15,  # 20-15

            ]
            if pain==0:
                color="blue"
            if pain == 1:
                color = "red"
            self.agent1 = self.canvas.create_polygon(self.points3, fill=color)
        
        s1 = self.canvas.coords(self.agent1)
        self.render()#显示当前的动作指令是什么

        self.centre1 = [(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2]
        if self.centre1[0] > (9 / 2) * 40:
            self.action_hurt1 = 0

        # whether hurt
        if self.action_hurt1 == 0:
            true_action = action
        else:
            if action == 0:
                true_action = 1
            if action == 1:
                true_action = 0
            if action == 2:
                true_action = 3
            if action == 3:
                true_action = 2
        

            
        base_action = np.array([0, 0])
        

       
        # predict next state
        self.centre1 = [(s1[4] + s1[8]) / 2, (s1[5] + s1[9]) / 2]
        pre_displacement1 = np.array([0, 0])
        if self.centre1[0] <= ((MAZE_H - 1) / 2 + 1) * UNIT:  # 120
            if action == 0:  # up
                if self.centre1[1] > UNIT:
                    pre_displacement1 = np.array([0, -40])
            elif action == 1:  # down
                if self.centre1[1] < (MAZE_W - 1) * UNIT:
                    pre_displacement1 = np.array([0, 40])
            elif action == 2:  # right
                if self.centre1[0] < ((MAZE_H - 1) / 2 - 1) * UNIT:
                    pre_displacement1 = np.array([40, 0])
            elif action == 3:  # left
                if self.centre1[0] > UNIT:
                    pre_displacement1 = np.array([-40, 0])
        else:
            if action == 0:  # up
                if self.centre1[1] > UNIT:
                    pre_displacement1 = np.array([0, -40])
            elif action == 1:  # down
                if self.centre1[1] < (MAZE_W - 1) * UNIT:
                    pre_displacement1 = np.array([0, 40])
            elif action == 2:  # right
                if self.centre1[0] < (MAZE_H - 1) * UNIT:
                    pre_displacement1 = np.array([40, 0])
            elif action == 3:  # left
                if self.centre1[0] > ((MAZE_H - 1) / 2 + 2) * UNIT:
                    pre_displacement1 = np.array([-40, 0])
        
        
        
        
        
        # true next state
        displacement1 = np.array([0, 0])
        
        if self.centre1[0] <= ((MAZE_H - 1) / 2 + 1) * UNIT:
            if true_action == 0:  # up
                if self.centre1[1] > UNIT:
                    displacement1=np.array([0,-40])
            elif true_action == 1:  # down
                if self.centre1[1] < (MAZE_W - 1) * UNIT:
                    displacement1=np.array([0,40])
            elif true_action == 2:  # right
                if self.centre1[0] < ((MAZE_H - 1) / 2 - 1) * UNIT:
                    displacement1=np.array([40,0])
            elif true_action == 3:  # left
                if self.centre1[0] > UNIT:
                    displacement1=np.array([-40,0])
        else:
            if true_action == 0:  # up
                if self.centre1[1] > UNIT:
                    displacement1=np.array([0,-40])
            elif true_action == 1:  # down
                if self.centre1[1] < (MAZE_W - 1) * UNIT:
                    displacement1=np.array([0,40])
            elif true_action == 2:  # right
                if self.centre1[0] < (MAZE_H - 1) * UNIT:
                    displacement1=np.array([40,0])
            elif true_action == 3:  # left
                if self.centre1[0] > ((MAZE_H - 1) / 2 + 2) * UNIT:
                    displacement1=np.array([-40,0])
        self.canvas.move(self.agent1, displacement1[0], displacement1[1])
        s1_ = self.canvas.coords(self.agent1)
        sss = [(s1_[4] + s1_[8]) / 2, (s1_[5] + s1_[9]) / 2]


        return displacement1, pre_displacement1,s1_,sss


    def agent_help(self):
        s = self.canvas.coords(self.agent)
        self.centre2= [(s[4] + s[8]) / 2, (s[5] + s[9]) / 2]
        if all(self.centre2 == self.oval_center):
            self.canvas.delete(self.wall[3])
            self.render()
            self.open_door=1
        else:    
            self.canvas.move(self.agent, -40, 0)  # move agent
            self.render()
            self.canvas.move(self.agent, -40, 0)
            self.render()
            self.canvas.move(self.agent, -40, 0)
            self.render()
            self.canvas.move(self.agent, 0, 40)
            self.render()
        s_ = self.canvas.coords(self.agent)  # next state
        
        return s_

    def _set_danger(self):
        hell1_center = np.array([140, 60])
        self.hell1 = self.canvas.create_agent1angle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        hell2_center = np.array([100, 140])
        self.hell2 = self.canvas.create_agent1angle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        # self.canvas.create_bitmap((40 , 40), bitmap='error')
        self.canvas.pack()
        self.danger=1


    def _set_wall(self):
        wall_center=[]
        self.wall=[]
        for i in range(MAZE_W):
            wall_center.append([])
            self.wall.append([])
        for i in range(MAZE_W):
            wall_center[i]=np.array([(MAZE_H*UNIT)/2,((i)*UNIT)+UNIT/2])
            self.wall[i] = self.canvas.create_agent1angle(
                wall_center[i][0] - 20, wall_center[i][1] - 20,
                wall_center[i][0] + 20, wall_center[i][1] + 20,
                fill='grey')
        self.canvas.pack()

    
    def generate_expression1(self,pain1):
        if pain1==1:
            self.canvas.itemconfig(self.agent1, fill="red", outline='black')
            self.canvas.pack()
        if pain1 ==0:
            self.canvas.itemconfig(self.agent1, fill="blue", outline='black')
            self.canvas.pack()
    def render(self):
        time.sleep(0.2)
        self.update()




