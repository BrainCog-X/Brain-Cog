import numpy as np
np.random.seed(1)
import tkinter as tk
import time
from PIL import ImageGrab

UNIT = 40   # pixels
MAZE_H = 9  # grid height
MAZE_W = 4 # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('self-pain')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self._build_maze()
        self.danger=0
        self.action_hurt=0
        self.sensory_hurt = 0
        self.open_door = 0
        self.pain_state=0

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


        # create agent
        self.orgin=[20,20]
        # 上
        self.points0 = [
            # 右下
            self.orgin[0]+15,#35
            self.orgin[1]+15,#35
            # 左下
            self.orgin[0]-15,#5
            self.orgin[1]+15,#35
            # 左上+
            self.orgin[0]-15,#5
            self.orgin[1],#20
            # 顶点
            self.orgin[0],#20
            self.orgin[1]-15,#5
            # 右上+
            self.orgin[0]+15,#35
            self.orgin[1],#20
        ]
        # self.rect0 = self.canvas.create_polygon(self.points0, fill="green")
        # self.agent_action0 = self.canvas.coords(self.rect0)

        # 下
        self.points1 = [
            # 左上
            self.orgin[0]-15,#5
            self.orgin[1]-15,#5
            # 右上
            self.orgin[0]+15,#35
            self.orgin[1]-15,#5
            # 右下+
            self.orgin[0]+15,#35
            self.orgin[1],#20
            # 顶点
            self.orgin[0],#20
            self.orgin[1]+15,#35
            # 左下+
            self.orgin[0]-15,#5
            self.orgin[1],#20
        ]
        self.rect = self.canvas.create_polygon(self.points1, fill="green")
        # self.agent_action1 = self.canvas.coords(self.rect1)

        # 右
        self.points2 = [
            # 左下
            self.orgin[0]-15,#5
            self.orgin[1]+15,#35
            # 左上
            self.orgin[0]-15,#5
            self.orgin[1]-15,#5
            # 右上+
            self.orgin[0],#20
            self.orgin[1]-15,#5
            # 顶点
            self.orgin[0]+15,#35
            self.orgin[1],#20
            # 右下+
            self.orgin[0],#20
            self.orgin[1]+15,#35
        ]
        # self.rect2 = self.canvas.create_polygon(self.points2, fill="green")
        # self.agent_action2 = self.canvas.coords(self.rect2)

        # 左
        self.points3 = [
            # 右上
            self.orgin[0]+15,#20+15
            self.orgin[1]-15,#20-15
            # 右下
            self.orgin[0]+15,#20+15
            self.orgin[1]+15,#20+15
            # 左下+
            self.orgin[0],#20
            self.orgin[1]+15,#20+15
            # 顶点
            self.orgin[0]-15,#20-15
            self.orgin[1],#20
            # 左上+
            self.orgin[0],#20
            self.orgin[1]-15,#20-15


        ]
        # self.rect3 = self.canvas.create_polygon(self.points3, fill="green")
        # self.agent_action3 = self.canvas.coords(self.rect3)
        self.canvas.pack()


    #reset agent location
    def reset(self):
        self.open_door = 0
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        self.orgin = [20, 20]
        # 下
        self.points1 = [
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
        self.rect = self.canvas.create_polygon(self.points1, fill="green")
        # self.agent_action1 = self.canvas.coords(self.rect1)
        return self.canvas.coords(self.rect)

    def step(self, s, action, pain):
        s = self.canvas.coords(self.rect)
        self.centre = [(s[4] + s[8]) / 2, (s[5] + s[9]) / 2]

        # danger or switch
        if self.danger==1:
            if all(self.centre == self.oval_center):
                s_color = 'yellow'
                self.canvas.delete(self.wall[3])
                self.render()
                # self.getter(self.canvas)
                self.render()
                # self.getter(self.canvas)#figure8 ,figure3.1 all red changed to green
                self.open_door = 1

                move = np.array([80, 0])
                self.canvas.move(self.rect, move[0], move[1])

                s = self.canvas.coords(self.rect)
                self.render()
                # self.getter(self.canvas)
            elif all(self.centre == self.hell1_center):
                s_color = 'black'
                self.action_hurt = 1
                self.render()
                # self.getter(self.canvas)#figure4
                self.render()
            else:
                s_color = 'white'




        # modify current state
        self.canvas.delete(self.rect)# 主要为开关那几步考虑，所以重复写了
        self.centre = [(s[4] + s[8]) / 2, (s[5] + s[9]) / 2]

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
                color="green"
            if pain == 1:
                color = "red"
            self.rect = self.canvas.create_polygon(self.points0, fill=color)
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
                color="green"
            if pain == 1:
                color = "red"
            self.rect = self.canvas.create_polygon(self.points1, fill=color)
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
                color="green"
            if pain == 1:
                color = "red"
            self.rect = self.canvas.create_polygon(self.points2, fill=color)
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
                color="green"
            if pain == 1:
                color = "red"
            self.rect = self.canvas.create_polygon(self.points3, fill=color)
        s = self.canvas.coords(self.rect)
        self.render()#显示当前的动作指令是什么
        # self.getter(self.canvas)#figure5 after figure4


        if s[0] > (9 / 2) * 40:
            self.action_hurt = 0
        # ensure ture action
        base_action = np.array([0, 0])
        if self.action_hurt == 0:
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

        # predict next state
        b = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        if self.centre[0] <= ((MAZE_H - 1) / 2 +1) * UNIT:#120
            if action == 0:  # up
                if self.centre[1] > UNIT:
                    b = [0, -40, 0, -40,0, -40, 0, -40,0, -40]
            elif action == 1:  # down
                if self.centre[1] < (MAZE_W - 1) * UNIT:
                    b = [0, 40, 0, 40,0, 40, 0, 40, 0, 40]
            elif action == 2:  # right
                if self.centre[0] < ((MAZE_H - 1) / 2 - 1) * UNIT:
                    b = [40, 0, 40, 0,40, 0, 40, 0,40, 0]
            elif action == 3:  # left
                if self.centre[0] > UNIT:
                    b = [-40, 0, -40, 0,-40, 0, -40, 0,-40, 0]
        else:
            if action == 0:  # up
                if self.centre[1] > UNIT:
                    b = [0, -40, 0, -40,0, -40, 0, -40,0, -40]
            elif action == 1:  # down
                if self.centre[1] < (MAZE_W - 1) * UNIT:
                    b = [0, 40, 0, 40,0, 40, 0, 40, 0, 40]
            elif action == 2:  # right
                if self.centre[0] < (MAZE_H - 1) * UNIT:
                    b = [40, 0, 40, 0,40, 0, 40, 0,40, 0]
            elif action == 3:  # left
                if self.centre[0] > ((MAZE_H - 1) / 2 + 2) * UNIT:
                    b = [-40, 0, -40, 0,-40, 0, -40, 0,-40, 0]
        s_predict = []
        for i in range(len(b)):
            s_predict1 = s[i] + b[i]
            s_predict.append(s_predict1)


        # true next state
        if self.centre[0]<=((MAZE_H - 1) / 2 +1) * UNIT:
            if true_action == 0:  # up
                if self.centre[1] > UNIT:
                    base_action[1] -= UNIT
            elif true_action == 1:  # down
                if self.centre[1] < (MAZE_W - 1) * UNIT:
                    base_action[1] += UNIT
            elif true_action == 2:  # right
                if self.centre[0] < ((MAZE_H - 1) / 2 - 1) * UNIT:
                    base_action[0] += UNIT
            elif true_action == 3:  # left
                if self.centre[0] > UNIT:
                    base_action[0] -= UNIT
        else:
            if true_action == 0:  # up
                if self.centre[1] > UNIT:
                    base_action[1] -= UNIT
            elif true_action == 1:  # down
                if self.centre[1] < (MAZE_W - 1) * UNIT:
                    base_action[1] += UNIT
            elif true_action == 2:  # right
                if self.centre[0] < (MAZE_H - 1) * UNIT:
                    base_action[0] += UNIT
            elif true_action == 3:  # left
                if self.centre[0] > ((MAZE_H - 1) / 2 + 2) * UNIT:
                    base_action[0] -= UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.rect)

        return s_, s_predict, s_color

    def step_RL1(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        if s[0] <= ((MAZE_H - 1) / 2 + 1) * UNIT:
            if action == 0:  # up
                if s[1] > UNIT:
                    base_action[1] -= UNIT
            elif action == 1:  # down
                if s[1] < (MAZE_W - 1) * UNIT:
                    base_action[1] += UNIT
            elif action == 2:  # right
                if s[0] < ((MAZE_H - 1) / 2 - 1) * UNIT:
                    base_action[0] += UNIT
            elif action == 3:  # left
                if s[0] > UNIT:
                    base_action[0] -= UNIT
        else:
            if action == 0:  # up
                if s[1] > UNIT:
                    base_action[1] -= UNIT
            elif action == 1:  # down
                if s[1] < (MAZE_W - 1) * UNIT:
                    base_action[1] += UNIT
            elif action == 2:  # right
                if s[0] < (MAZE_H - 1) * UNIT:
                    base_action[0] += UNIT
            elif action == 3:  # left
                if s[0] > ((MAZE_H - 1) / 2 + 2) * UNIT:
                    base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)  # next state

        if s_==self.canvas.coords(self.hell1):
            self.canvas.itemconfig(self.rect, fill="red", outline='red')
            reward = -1
            self.pain_state=1
        else:
            reward = 0
        return s_, reward,self.pain_state

    def step_RL2(self, action):
        s = self.canvas.coords(self.rect)
        if s == self.canvas.coords(self.oval):
            self.canvas.delete(self.wall[3])
            self.open_door = 1
            move = np.array([40, 0])
            self.canvas.move(self.rect, move[0], move[1])
            move = np.array([40, 0])
            self.canvas.move(self.rect, move[0], move[1])
            self.render()
            self.canvas.itemconfig(self.rect, fill="green", outline='green')
            self.render()

        base_action = np.array([0, 0])

        if s[0] <= ((MAZE_H - 1) / 2 + 1) * UNIT:
            if action == 0:  # up
                if s[1] > UNIT:
                    base_action[1] -= UNIT
            elif action == 1:  # down
                if s[1] < (MAZE_W - 1) * UNIT:
                    base_action[1] += UNIT
            elif action == 2:  # right
                if s[0] < ((MAZE_H - 1) / 2 - 1) * UNIT:
                    base_action[0] += UNIT
            elif action == 3:  # left
                if s[0] > UNIT:
                    base_action[0] -= UNIT
        else:
            if action == 0:  # up
                if s[1] > UNIT:
                    base_action[1] -= UNIT
            elif action == 1:  # down
                if s[1] < (MAZE_W - 1) * UNIT:
                    base_action[1] += UNIT
            elif action == 2:  # right
                if s[0] < (MAZE_H - 1) * UNIT:
                    base_action[0] += UNIT
            elif action == 3:  # left
                if s[0] > ((MAZE_H - 1) / 2 + 2) * UNIT:
                    base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)  # next state

        if s_ == self.canvas.coords(self.oval):


            self.open_door = 1

            if self.pain_state == 0:
                reward = 0
            if self.pain_state == 1:
                reward = 1
                self.pain_state = 0

        elif s_ == self.canvas.coords(self.hell1):
            self.canvas.itemconfig(self.rect, fill="red", outline='red')
            reward = -1
            self.pain_state = 1
            self.render()

        else:
            reward = 0

        return s_, reward, self.pain_state

    def _set_danger(self):
        self.hell1_center = np.array([60, 60])
        self.hell1 = self.canvas.create_oval(
            self.hell1_center[0] - 15, self.hell1_center[1] - 15,
            self.hell1_center[0] + 15, self.hell1_center[1] + 15,
            fill='black')
        # self.canvas.create_bitmap((40 , 40), bitmap='error')
        self.hell = self.canvas.coords(self.hell1)
        self.canvas.pack()
        self.danger=1

    def _set_switch(self):
        self.oval_center = np.array([(MAZE_H * UNIT) / 2 - UNIT, ((MAZE_W + 4) * UNIT) / 2 - UNIT / 2])
        self.oval = self.canvas.create_oval(
            self.oval_center[0] - 15, self.oval_center[1] - 15,
            self.oval_center[0] + 15, self.oval_center[1] + 15,
            fill='yellow')
        self.switch = self.canvas.coords(self.oval)
        self.canvas.pack()


    def _set_wall(self):
        wall_center=[]
        self.wall=[]
        for a in range(MAZE_W):
            wall_center.append([0,0])
            self.wall.append([])
        for b in range(MAZE_W):
            wall_center[b]=np.array([(MAZE_H*UNIT)/2,((b)*UNIT)+UNIT/2])
            self.wall[b] = self.canvas.create_rectangle(
                wall_center[b][0] - 20, wall_center[b][1] - 20,
                wall_center[b][0] + 20, wall_center[b][1] + 20,
                fill='grey')
        self.wall0 = self.canvas.coords(self.wall[0])
        self.wall1 = self.canvas.coords(self.wall[1])
        self.wall2 = self.canvas.coords(self.wall[2])
        self.wall3 = self.canvas.coords(self.wall[3])

        # self.canvas.pack()

    def generate_expression(self,pain):
        if pain==1:
            self.canvas.itemconfig(self.rect, fill="red", outline='red')
            # self.canvas.pack()
        if pain == 0:
            self.canvas.itemconfig(self.rect, fill="green", outline='green')
            # self.canvas.pack()

    def render(self):
        time.sleep(0.01)
        self.update()

    # def getter(self, widget):
    #     widget.update()
    #     x = tk.Tk.winfo_rootx(self) + widget.winfo_x()
    #     y = tk.Tk.winfo_rooty(self) + widget.winfo_y()
    #     x1 = x + widget.winfo_width()
    #     y1 = y + widget.winfo_height()
    #     ImageGrab.grab().crop((x, y, x1, y1)).save("first.jpg")
    #     return ImageGrab.grab().crop((x, y, x1, y1))



