"""
Zoe Zhao 2022.5
Env Demo
"""

from numpy import argmax
import random, time, pygame, sys
import pygame
pygame.init()
from pygame.locals import *
# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"

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
class FalseBelief_env1(object):
    def __init__(self, reward=10):
        super(FalseBelief_env1, self).__init__()
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
        self.x = 0
        self.n_features = 30
        self.reward = reward

    def _build_AB(self):
        global FPSCLOCK, DISPLAYSURF, BASICFONT, BIGFONT
        pygame.init()
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
        state2 = np.r_[b, np.ones((4, 5))].astype(np.int)
        statea = np.r_[c, np.ones((4, 5))].astype(np.int)
        NPC_2_state = state2
        Agent_state = statea

        NPC_2_state[self.NPC_2['y']-1, self.NPC_2['x']-1] = observer
        r = shelter_env(NPC_2_state[:5, :])
        NPC_2_state[:5, :] = shelter_env(NPC_2_state[:5, :])

        Agent_state[self.agent['y']-1, self.agent['x']-1] = observer
        p = shelter_env(Agent_state[:5, :])
        Agent_state[:5, :] = shelter_env(Agent_state[:5, :])


        self.NPC_2['obs'] = r
        self.NPC_2['obs'] = self.gain_obs(self.NPC_2['obs'], NPC_2_state, self.agent, 4)
        self.NPC_2['axis'] = self.gain_axis(self.NPC_2, NPC_2_state, 6, self.agent, 2, 4)

        self.agent['obs'] = p
        self.agent['obs'] = self.gain_obs(self.agent['obs'], Agent_state,  self.NPC_2, 3)
        self.agent['axis'] = self.gain_axis(self.agent, Agent_state, 6, self.NPC_2, 2, 3)

        return NPC_2_state, Agent_state#NPC_1_state,

    def gain_obs(self, a,aa,c,cc):
        if aa[c['y']-1, c['x']-1] ==1:
            a[c['y'] - 1, c['x'] - 1] = cc

        return a

    def gain_axis(self,a,aa,b,c,bb,cc):
        axis = []
        axis.append([a['y'], a['x']])
        if b == 6:
            axis.append([6, 6])
        else:
            axis.append([6, 6])
        if aa[c['y']-1, c['x']-1] != 0:
            axis.append([c['y'] , c['x']])
        else:
            axis.append([6, 6])
        return axis


    def interact(self, action_NPC2, action_agent):
        self.agent['reward'] = 0
        self.NPC_2['reward'] = 0
        #三个智能体分别会看到什么？
        NPC_2_state, Agent_state = self.actu_obs()# NPC_1_state,
        #看到这些状态，智能体们会分别采取什么行为？ ---depend on RL
        #这些行为对状态的影响  ---首先，影响本身的位置坐标，然后,影响观测
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
                self.NPC_2['reward'] = (((dis1 - dis2) * 10 - 1 / 2) / dis1)
                while self.NPC_2['reward'] < 0.5 and self.NPC_2['reward'] > -0.5:
                    self.NPC_2['reward'] = self.NPC_2['reward'] * 2
                if self.NPC_2['reward'] > 1:
                    self.NPC_2['reward'] = 1
                elif self.NPC_2['reward'] < -1:
                    self.NPC_2['reward'] = -1
            else:
                self.NPC_2['reward'] = -0.9  # * (1 / dis1)

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
            #
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

        NPC_2_state, Agent_state = self.actu_obs()

        #判断是否会相撞?
        location = [(self.NPC_2['x'], self.NPC_2['y']),\
                    (self.agent['x'], self.agent['y'])]

        #达到目标或者相撞都会结束该智能体的回合
        terminal = self.gameover(location)
        if self.agent['Done'] == False and terminal[1] == True:
            self.agent['Done'] = True
            self.agent['reward'] = -self.reward
            self.agent['color'] = Red
        if self.NPC_2['Done'] == False and terminal[0] == True:
            self.NPC_2['Done'] = True
            self.NPC_2['reward'] = -self.reward
            self.NPC_2['color'] = Red

        return  NPC_2_state, Agent_state

    def SHOW(self):
        DISPLAYSURF.fill(BGColor)
        self.DrawBoard(self.board)
        # self.DrawPiece(self.NPC_1)
        self.DrawPiece(self.NPC_2)
        self.DrawPiece(self.agent)
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        # return flag

    def reset(self):
        self._agent_init()

    def getBlankBoard(self):
        # board = data_transfer('env_1.txt','env_11.txt')
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
        terminal = [False, False]   #NPC_2, agent
        for r in range(len(location) - 1):
            for c in range(r + 1, len(location)):
                if location[r] == location[c]:
                    result = True       ####相撞会带来一个巨大的副奖励，并且结束该回合
                    boom = location[r] ###############相撞################
                if result == True:
                    terminal[r] = True
                    terminal[c] = True
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
            pygame.draw.rect(DISPLAYSURF, (205, 155, 155	), (xpixel + 1, ypixel + 1, BoxSize - 1, BoxSize - 1))
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

    def DrawCircle(self,  xbox, ybox, color, xpixel=None, ypixel=None):
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


