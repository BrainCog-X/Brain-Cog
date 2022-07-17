
from numpy import argmax
import random, time, pygame, sys
import pygame
pygame.init()
from pygame.locals import *
import os
# os.environ['SDL_AUDIODRIVER'] = 'dsp'
# os.environ['SDL_VIDEODRIVER']='windib'
os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.environ['DISPLAY'] = "localhost:13.0"

from rulebasedpolicy.world_model import *
from rulebasedpolicy.statedata_pre import *
from rulebasedpolicy.Find_a_way import *
import numpy as np
from utils.one_hot import one_hot

# =============================================================================
# set the value of interface
# =============================================================================
FPS = 25
WinWidth = 340 #window width
WinHeight = 260 #window width
BoxSize = 20    #the size of one grid
GridWidth = 7   #the number of lattices are there in the x-axis
GridHeight = 7  #the number of lattices are there in the y-axis
#representation of different objective
BlankBox = 1
Wall = 5
Obstacle = 5
observer = 8
obeservation_1 = 11
obeservation_2 = 22
obeservation_3 = 33
#Text = None
XMargin = int((WinWidth - GridWidth * BoxSize)/2)
TopMargin = int((WinHeight - GridHeight * BoxSize))/2-5
# =============================================================================
# set color
# =============================================================================
White = (255,255,255)
Gray = (185,185,185)
Black = (0,0,0)
Red = (200,0,0)
Green = (0,139,0)
Green_B = (78, 238, 148)
Light_A = (233, 232, 170)
Blue = (30, 144, 255)
pink = (238, 99, 99)
BoardColor = White
BGColor = White
TextColor = White
Test = []
# =============================================================================
# agents - env interactive
# =============================================================================
class FalseBelief_env(object):
    def __init__(self, reward=10):
        super(FalseBelief_env, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right', 'stay']
        self.action_move = {
            0 : (0, -1),
            1 : (0, 1),
            2 : (-1, 0),
            3 : (1, 0),
            4 :(0, 0)
        }#[(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
        self.n_actions = len(self.action_space)
        self._build_AB()
        self.board, self.obs = self.getBlankBoard()
        self._agent_init()
        self.score = 0
        self.steps = 0
        self.n = 0
        self.R = int(5/2) * (BoxSize - 5)
        self.trigger = 0
        self.x = 0
        self.n_features = 30
        self.reward = reward

    def _build_AB(self):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, BIGFONT
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WinWidth, WinHeight))
        BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
        BIGFONT = pygame.font.Font('freesansbold.ttf', 100)
        pygame.display.set_caption('AB')
        pygame.display.update()
        FPSCLOCK.tick()

    def _agent_init(self):
        """
        Aim:Initialize the basic information of the agent
        """
        self.NPC_1 = {
            'shape' : [['#']],
            'x' : 3, #row
            'y' : 1, #column
            'color' : Blue,
            'style' : "circle",
            'obs' : None,
            'axis' : None,#,[[1,3],[3,5],[4,2]]
            'reward' : 0,
            'Done' : False
        }
        self.NPC_2 = {
            'shape' : [['@']],
            'x' : 5, #row
            'y' : 3, #column
            'color' : pink,
            'style' : "circle",
            'obs' : None,
            'axis' :None ,#,[[3,5],[1,3],[4,2]]
            'reward': 0,
            'Done': False
        }
        self.agent = {
            'shape' : [['$']],
            'x' : 2, #row
            'y' : 4, #column
            'color' : Green_B,
            'style' : "circle",
            'obs' : None,
            'axis' : None,#[[4,2],[1,3],[3,5]]
            'reward': 0,
            'Done': False
        }

    def actu_obs(self):
        """
        将状态转化成可以训练的数据形式
        """
        _, state = self.getBlankBoard()
        a = state
        b = state
        c = state
        state1 = np.r_[a, np.ones((4, 5))].astype(np.int_)
        state2 = np.r_[b, np.ones((4, 5))].astype(np.int_)
        statea = np.r_[c, np.ones((4, 5))].astype(np.int_)
        NPC_1_state = state1
        NPC_2_state = state2
        Agent_state = statea

        NPC_1_state[self.NPC_1['y']-1, self.NPC_1['x']-1] = observer
        q = shelter_env(NPC_1_state[:5, :])
        NPC_1_state[:5, :] = shelter_env(NPC_1_state[:5, :])


        NPC_2_state[self.NPC_2['y']-1, self.NPC_2['x']-1] = observer
        r = shelter_env(NPC_2_state[:5, :])
        NPC_2_state[:5, :] = shelter_env(NPC_2_state[:5, :])

        Agent_state[self.agent['y']-1, self.agent['x']-1] = observer
        p = shelter_env(Agent_state[:5, :])
        Agent_state[:5, :] = shelter_env(Agent_state[:5, :])
        """
        ########### num ############
        #2-NPC1 in other agents' obs
        #3-NPC2 in other agents' obs
        #4-Agent in other agents' obs        
        """
        self.NPC_1['obs'] = q
        self.NPC_1['obs'] = self.gain_obs(self.NPC_1['obs'],NPC_1_state,self.NPC_2,self.agent,3,4)
        self.NPC_1['axis'] = self.gain_axis(self.NPC_1,NPC_1_state,self.NPC_2,self.agent,3,4)

        self.NPC_2['obs'] = r
        self.NPC_2['obs'] = self.gain_obs(self.NPC_2['obs'], NPC_2_state, self.NPC_1, self.agent, 2, 4)
        self.NPC_2['axis'] = self.gain_axis(self.NPC_2, NPC_2_state, self.NPC_1, self.agent, 2, 4)

        self.agent['obs'] = p
        self.agent['obs'] = self.gain_obs(self.agent['obs'], Agent_state, self.NPC_1, self.NPC_2, 2, 3)
        self.agent['axis'] = self.gain_axis(self.agent, Agent_state, self.NPC_1, self.NPC_2, 2, 3)

        return NPC_1_state, NPC_2_state, Agent_state

    def gain_obs(self, a, aa, b, c, bb, cc):
        """
        获得智能体真正的环境遮挡关系
        @param a: self - observation
        @param aa: self - self-axis, other-b-axis, other-c-axis
        @param b: other-b 遮挡后的可见区域 5*5
        @param c: other-c 遮挡后的可见区域 5*5
        @param bb: other-b' num
        @param cc: other-c' num
        @return: self - observation
        """
        if aa[b['y']-1, b['x']-1] == 1:
            a[b['y']-1, b['x']-1] = bb

        if aa[c['y']-1, c['x']-1] ==1:
            a[c['y'] - 1, c['x'] - 1] = cc

        return a

    def gain_axis(self,a,aa,b,c,bb,cc):
        """
        获得坐标，但是看不见的坐标就用6来表示
        @param a:
        @param aa:
        @param b:
        @param c:
        @param bb:
        @param cc:
        @return:
        """
        axis = []
        axis.append([a['y'], a['x']])
        if aa[b['y']-1, b['x']-1] != 0:
            axis.append([b['y'], b['x']])
        else:
            axis.append([6,6])
        if aa[c['y']-1, c['x']-1] != 0:
            axis.append([c['y'] , c['x']])
        else:
            axis.append([6, 6])
        return axis

    def interact(self, action_NPC1, action_NPC2, action_agent):
        """
        三个智能体进行交互
        @param action_NPC1: action
        @param action_NPC2: actionF
        @param action_agent: action
        @return:5*5 NPC1遮挡后看见了什么 5*5 NPC2遮挡后看见了什么 5*5 agent遮挡后看见了什么
        """
        self.agent['reward'] = 0
        self.NPC_1['reward'] = 0
        self.NPC_2['reward'] = 0
        #三个智能体分别会看到什么？
        NPC_1_state, NPC_2_state, Agent_state = self.actu_obs()

        #看到这些状态，智能体们会分别采取什么行为？ ---depend on RL
        #这些行为对状态的影响  ---首先，影响本身的位置坐标，然后,影响观测
        base = np.where(np.array(self.board) == obeservation_1)
        base_x = int(base[0])
        base_y= int(base[1])
        if self.NPC_1['Done'] == False:
            dis1 = np.sqrt(np.square(base_x - self.NPC_1['y']) + np.square(base_y - self.NPC_1['x']))
            if self.isNotWall(self.board, self.NPC_1, self.action_move[action_NPC1][0], \
                              self.action_move[action_NPC1][1]):
                self.NPC_1['x'] = self.NPC_1['x'] + self.action_move[action_NPC1][0]
                self.NPC_1['y'] = self.NPC_1['y'] + self.action_move[action_NPC1][1]
                dis2 = np.sqrt(np.square(base_x - self.NPC_1['y']) + np.square(base_y - self.NPC_1['x']))
                self.NPC_1['reward'] = (((dis1 - dis2) * 2 - 1) / dis1)
            else:
                self.NPC_1['reward'] = -1 * (1 / dis1)

            if self.board[self.NPC_1['y'], self.NPC_1['x']] == obeservation_1:
                self.NPC_1['reward'] = 50
                self.NPC_1['Done'] = True

        base = np.where(np.array(self.board) == obeservation_2)
        base_x = int(base[0])
        base_y= int(base[1])
        if self.NPC_2['Done'] == False:
            dis1 = np.sqrt(np.square(base_x - self.NPC_2['y']) + np.square(base_y - self.NPC_2['x']))
            if self.isNotWall(self.board, self.NPC_2, self.action_move[action_NPC2][0], \
                              self.action_move[action_NPC2][1]):
                self.NPC_2['x'] = self.NPC_2['x'] + self.action_move[action_NPC2][0]
                self.NPC_2['y'] = self.NPC_2['y'] + self.action_move[action_NPC2][1]
                dis2 = np.sqrt(np.square(base_x - self.NPC_2['y']) + np.square(base_y - self.NPC_2['x']))
                self.NPC_2['reward'] = (((dis1 - dis2)*10 - 1/2) / dis1)
                while self.NPC_2['reward'] < 0.5 and self.NPC_2['reward'] > -0.5 :
                    self.NPC_2['reward'] = self.NPC_2['reward'] * 2
                if self.NPC_2['reward'] > 1:
                    self.NPC_2['reward'] = 1
                elif self.NPC_2['reward'] < -1:
                    self.NPC_2['reward'] = -1

            else:
                self.NPC_2['reward'] = -0.9 #* (1 / dis1)

            if self.board[self.NPC_2['y'], self.NPC_2['x']] == obeservation_2:
                self.NPC_2['reward'] = self.reward
                self.NPC_2['Done'] = True

        base = np.where(np.array(self.board) == obeservation_3)
        base_x = int(base[0])
        base_y= int(base[1])
        if self.agent['Done'] == False:
            dis1 = np.sqrt(np.square(base_x - self.agent['y']) + np.square(base_y - self.agent['x']))
            if self.isNotWall(board=self.board, piece=self.agent, xT=self.action_move[action_agent][0], \
                              yT=self.action_move[action_agent][1]):
                self.agent['x'] = self.agent['x'] + self.action_move[action_agent][0]
                self.agent['y'] = self.agent['y'] + self.action_move[action_agent][1]
                dis2 = np.sqrt(np.square(base_x - self.agent['y']) + np.square(base_y - self.agent['x']))
            #     self.agent['reward'] = (((dis1 - dis2) * 2 - 1) / dis1)
            # else:
            #     # print('action', action_agent)
            #     self.agent['reward'] = -1 * (1 / dis1)
                self.agent['reward'] = (((dis1 - dis2)*10 - 1/2) / dis1)
                while self.agent['reward'] < 0.5 and self.agent['reward'] > -0.5 :
                    self.agent['reward'] = self.agent['reward'] * 2 - 0.1
                if self.agent['reward'] > 1:
                    self.agent['reward'] = 1
                elif self.agent['reward'] < -1:
                    self.agent['reward'] = -1
            else:
                self.agent['reward'] = -0.9 #* (1 / dis1)
            if self.board[self.agent['y'], self.agent['x']] == obeservation_3:
                self.agent['reward'] = self.reward
                self.agent['Done'] = True
        NPC_1_state, NPC_2_state, Agent_state = self.actu_obs()

        #判断是否会相撞?
        location = [(self.NPC_1['x'], self.NPC_1['y']), (self.NPC_2['x'], self.NPC_2['y']),\
                    (self.agent['x'], self.agent['y'])]

        #达到目标或者相撞都会结束该智能体的回合
        terminal = self.gameover(location)

        if terminal[0] == True and self.NPC_1['Done'] == False:
            self.NPC_1['Done'] = True
            self.NPC_1['reward'] = -50
            self.NPC_1['color'] = Red
        if terminal[1] == True and self.NPC_2['Done'] == False:
            self.NPC_2['Done'] = True
            self.NPC_2['reward'] = -self.reward
            self.NPC_2['color'] = Red
        if terminal[2] == True and self.agent['Done'] == False:
            self.agent['Done'] = True
            self.agent['reward'] = -self.reward
            self.agent['color'] = Red

        return NPC_1_state, NPC_2_state, Agent_state

    def SHOW(self):
        """
        显示函数
        """
        DISPLAYSURF.fill(BGColor)
        self.DrawBoard(self.board)
        self.DrawPiece(self.NPC_1)
        self.DrawPiece(self.NPC_2)
        self.DrawPiece(self.agent)
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        # return flag

    def reset(self):
        self._agent_init()

    def getBlankBoard(self):
        """
        11 - NPC1-goal
        22 - NPC2-goal
        33 - Agent-goal
        @return:
        """
        board = np.array([[1,1,1,5,5],[22,1,1,5,5],[1,1,1,1,1],[1,1,1,1,33],[1,1,1,11,1]])
        state_init = np.array([[1,1,1,5,5],[1,1,1,5,5],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
        board = big_env(board)
        # print(board)
        #        for x in range(GridWidth):
        #            for y in range(GridHeight):
        #                print(board[x][y],x,y)
        return board, state_init

    def ValidPos(self, piece1, piece2, xT=0, yT=0):
        """
        to judge the next place vaild or not
        @param piece1:
        @param piece2:
        @param xT:
        @param yT:
        @return:
        """
        if piece1['x'] == (piece2['x'] + xT) and piece1['y'] == (piece2['y'] + yT):
            return True
        return False

    def isNotWall(self, board, piece, xT=0 , yT=0 ):
        """
        判断是否到达墙
        @param board: board
        @param piece: agent
        @param xT:
        @param yT:
        @return:
        """
        if board[piece['y'] + yT][piece['x'] + xT] == Wall:#############
            return False
        else:
            return True

    def gameover(self, location):
        """
        回合是否结束，以及奖励值
        @param location:目标位置
        """
        result = False
        terminal = [False, False, False]

        if location[0] == location[1]:
            terminal[0] = True
            terminal[1] = True
        if location[2] == location[1]:
            terminal[2] = True
            terminal[1] = True
        if location[2] ==location[0]:
            terminal[2] = True
            terminal[0] = True

        return terminal


    def pixel(self, xbox, ybox):
        return (XMargin + (xbox * BoxSize)), (TopMargin + (ybox * BoxSize))

    def DrawBox(self, xbox, ybox, color, xpixel=None, ypixel=None):
        if color == BlankBox:
            return
        elif color == obeservation_1:
            if xpixel == None and ypixel == None:
                xpixel, ypixel = self.pixel(xbox, ybox)
            pygame.draw.rect(DISPLAYSURF, (60,107,255), (xpixel + 1, ypixel + 1, BoxSize - 1, BoxSize - 1))
        elif color == obeservation_2:
            if xpixel == None and ypixel == None:
                xpixel, ypixel = self.pixel(xbox, ybox)
            pygame.draw.rect(DISPLAYSURF, (205, 155, 155), (xpixel + 1, ypixel + 1, BoxSize - 1, BoxSize - 1))
        elif color == obeservation_3:
            if xpixel == None and ypixel == None:
                xpixel, ypixel = self.pixel(xbox, ybox)
            pygame.draw.rect(DISPLAYSURF, (154, 205, 50), (xpixel + 1, ypixel + 1, BoxSize - 1, BoxSize - 1))
        elif color == Wall:
            if xpixel == None and ypixel == None:
                xpixel, ypixel = self.pixel(xbox, ybox)
            pygame.draw.rect(DISPLAYSURF, Gray, (xpixel + 1, ypixel + 1, BoxSize - 1, BoxSize - 1))
        else:
            if xpixel == None and ypixel == None:
                xpixel, ypixel = self.pixel(xbox, ybox)
            pygame.draw.rect(DISPLAYSURF, color, (xpixel + 1, ypixel + 1, BoxSize - 1, BoxSize - 1))

    def fun_trigger(self):
        xpixel, ypixel = self.pixel(3, 5)
        pygame.draw.line(DISPLAYSURF, Red, (xpixel + BoxSize, ypixel - BoxSize),
                         (xpixel + BoxSize, ypixel - 2*BoxSize), 5)

    def DrawCircle(self,  xbox, ybox, color, xpixel=None, ypixel=None):
        """
        画圆
        @param xbox:
        @param ybox:
        @param color:
        @param xpixel:
        @param ypixel:
        """
        pygame.draw.circle(DISPLAYSURF,
                           color,
                           (int(xpixel+BoxSize/2), int(ypixel+BoxSize/2)),
                           int(0.3 * self.R))

    def DrawPiece(self, piece, xpixel=None, ypixel=None):
        if xpixel == None and ypixel == None:
            xpixel, ypixel = self.pixel(piece['x'], piece['y'])
        if piece['style'] == "circle":
            self.DrawCircle(None, None, piece['color'], xpixel, ypixel)
        else:
            self.DrawBox(None, None, piece['color'], xpixel, ypixel)

    def DrawBoard(self, board):
        pygame.draw.rect(DISPLAYSURF, BoardColor,
                         (XMargin - 3, TopMargin - 7, (GridWidth * BoxSize) + 8, (GridHeight * BoxSize) + 8), 5)
        pygame.draw.rect(DISPLAYSURF, BGColor, (XMargin, TopMargin, GridWidth * BoxSize, GridHeight * BoxSize))
        for x in range(GridWidth):
            for y in range(GridHeight):
                self.DrawBox(y, x, board[x][y])
        if self.trigger == 1:
            self.fun_trigger()
    def ShowScore(self, score):
        scoreSurf = BASICFONT.render('Score : %s' % score, True, TextColor)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (WinWidth - 250, 20)
        DISPLAYSURF.blit(scoreSurf, scoreRect)

    def Terminal(self, piece1, piece2, piece1_old, piece2_old):
        # print(piece1['x'],piece1['y'],piece2_old[0],piece2_old[1],'/',piece2['x'],piece2['y'],piece1_old[0],piece1_old[1])
        # if self.steps == 1:# wrong!!!!
        if piece1['x'] == piece2_old[0] and piece1['y'] == piece2_old[1] and piece2['x'] == piece1_old[0] and piece2['y'] == piece1_old[1]:
            return 1
        else:
            return 2

    def Paint(self, board, piece, color):
        board[piece['x']][piece['y']] = color
        piece['color'] = color
        return board

# if __name__ == "__main__":
#     env0 = FalseBelief_env0()
#     action_agent = 0
#     action_NPC2 = 1
#     action_NPC1 = 4
#     for i in range(10):
#         if i > 8:
#             break
#         else:
#             env0.interact(action_NPC1, action_NPC2, action_agent)
#
#             env0.SHOW()
#             time.sleep(2)
#     pygame.quit()
