import numpy as np
import torch,os,sys
from torch import nn
from torch.nn import Parameter

import abc
import math
from abc import ABC

import torch.nn.functional as F
import matplotlib.pyplot as plt
#from BrainCog.base.strategy.surrogate import *
from braincog.base.node.node import IFNode, SimHHNode
from braincog.base.learningrule.STDP import STDP, MutliInputSTDP
from braincog.base.connection.CustomLinear import CustomLinear
from braincog.base.brainarea.basalganglia import basalganglia
#from braincog.model_zoo.bdmsnn import BDMSNN

import pygame
from pygame.locals import *
from collections import deque
from random import randint
#os.environ["SDL_VIDEODRIVER"] = "dummy"
class BDMSNN(nn.Module):
    def __init__(self, num_state, num_action, weight_exc, weight_inh, node_type):
        """
        定义BDM-SNN网络
        :param num_state: 状态个数
        :param num_action: 动作个数
        :param weight_exc: 兴奋性连接权重
        :param weight_inh: 抑制性连接权重
        """
        super().__init__()
        # parameters
        BG = basalganglia(num_state, num_action, weight_exc, weight_inh, node_type)
        dm_connection = BG.getweight()
        dm_mask = BG.getmask()
        # input-dlpfc
        con_matrix9 = torch.eye((num_state), dtype=torch.float)
        dm_connection.append(CustomLinear(weight_exc * con_matrix9, con_matrix9))
        dm_mask.append(con_matrix9)
        # gpi-th
        con_matrix10 = torch.eye((num_action), dtype=torch.float)
        dm_mask.append(con_matrix10)
        dm_connection.append(CustomLinear(weight_inh * con_matrix10, con_matrix10))
        # th-pm
        dm_mask.append(con_matrix10)
        dm_connection.append(CustomLinear(weight_exc * con_matrix10, con_matrix10))
        # dlpfc-th
        con_matrix11 = torch.ones((num_state, num_action), dtype=torch.float)
        dm_mask.append(con_matrix11)
        dm_connection.append(CustomLinear(0.2 * weight_exc * con_matrix11, con_matrix11))
        # pm-pm
        con_matrix3 = torch.ones((num_action, num_action), dtype=torch.float)
        con_matrix4 = torch.eye((num_action), dtype=torch.float)
        con_matrix5 = con_matrix3 - con_matrix4
        con_matrix5 = con_matrix5
        dm_mask.append(con_matrix5)
        dm_connection.append(CustomLinear(5 * weight_inh * con_matrix5, con_matrix5))
        # dlpfc thalamus pm +bg
        self.weight_exc = weight_exc
        self.num_subDM = 8
        self.connection = dm_connection
        self.mask = dm_mask
        self.node = BG.node
        self.node_type = node_type
        if self.node_type == "hh":
            self.node.extend([SimHHNode() for i in range(self.num_subDM - BG.num_subBG)])
            self.node[6].g_Na = torch.tensor(12)
            self.node[6].g_K = torch.tensor(3.6)
            self.node[6].g_L = torch.tensor(0.03)
        if self.node_type == "lif":
            self.node.extend([IFNode() for i in range(self.num_subDM - BG.num_subBG)])
        self.learning_rule = BG.learning_rule
        self.learning_rule.append(MutliInputSTDP(self.node[5], [self.connection[10], self.connection[12]]))  # gpi-丘脑
        self.learning_rule.append(MutliInputSTDP(self.node[6], [self.connection[11], self.connection[13]]))  # pm
        self.learning_rule.append(STDP(self.node[7], self.connection[9]))

        out_shape=[self.connection[0].weight.shape[1],self.connection[1].weight.shape[1],self.connection[2].weight.shape[1],self.connection[4].weight.shape[1],self.connection[3].weight.shape[1],self.connection[10].weight.shape[1],self.connection[11].weight.shape[1],self.connection[9].weight.shape[1]]
        self.out = []
        self.dw = []
        for i in range(self.num_subDM):
            self.out.append(torch.zeros((out_shape[i]), dtype=torch.float))
            self.dw.append(torch.zeros((out_shape[i]), dtype=torch.float))

    def forward(self, input):
        """
        根据输入得到网络的输出
        :param input: 输入
        :return: 网络的输出
        """
        self.out[7] = self.node[7](self.connection[9](input))
        self.out[0], self.dw[0] = self.learning_rule[0](self.out[7])
        self.out[1], self.dw[1] = self.learning_rule[1](self.out[7])
        self.out[2], self.dw[2] = self.learning_rule[2](self.out[7], self.out[3])
        self.out[3], self.dw[3] = self.learning_rule[3](self.out[1], self.out[2])
        self.out[4], self.dw[4] = self.learning_rule[4](self.out[0], self.out[3], self.out[2])
        self.out[5], self.dw[5] = self.learning_rule[5](self.out[4], self.out[7])
        self.out[6], self.dw[6] = self.learning_rule[6](self.out[5], self.out[6])
        br = ["StrD1", "StrD2", "STN", "Gpe", "Gpi", "thalamus", "PM", "DLPFC"]
        for i in range(self.num_subDM):
            if torch.max(self.out[i]) > 0 and self.node_type == "hh":
                self.node[i].n_reset()
            print("every areas:", br[i], self.out[i])
        return self.out[6], self.dw

    def UpdateWeight(self, i, s, num_action, dw):
        """
        更新网络中第i组连接的权重
        :param i:要更新的连接组索引
        :param s:传入状态
        :param dw:更新权重的量
        :return:
        """
        if self.node_type == "hh":
            self.connection[i].update(0.2 * self.weight_exc * dw)
            self.connection[i].weight.data[s, [s * num_action, s * num_action + 1]] /= (self.connection[i].weight.data[s, [s * num_action, s * num_action + 1]].float().max() + 1e-12)
            self.connection[i].weight.data[s, :] = self.connection[i].weight.data[s, :] * self.weight_exc
        if self.node_type == "lif":
            dw_mean = dw[s, [s * num_action, s * num_action + 1]].mean()
            dw_std = dw[s, [s * num_action, s * num_action + 1]].std()
            dw[s, [s * num_action, s * num_action + 1]] = (dw[s, [s * num_action,s * num_action + 1]] - dw_mean) / dw_std
            dw[s, :] = dw[s, :] * self.mask[i][s, :]
            self.connection[i].update(dw)
            self.connection[i].weight.data[s, [s * num_action, s * num_action + 1]] /= (self.connection[i].weight.data[s, [s * num_action, s * num_action + 1]].float().max() + 1e-12)
        if i in [0, 1, 2, 6, 7, 11, 12]:
            self.connection[i].weight.data = torch.clamp(self.connection[i].weight.data, 0, None)
        if i in [3, 4, 5, 8, 10]:
            self.connection[i].weight.data = torch.clamp(self.connection[i].weight.data, None, 0)

    def reset(self):
        """
        reset神经元或学习法则的中间量
        :return: None
        """
        for i in range(self.num_subDM):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()

    def getweight(self):
        """
        获取网络的连接(包括权值等)
        :return: 网络的连接
        """
        return self.connection

def load_images():
    """Load all images required by the game and return a dict of them.

    The returned dict has the following keys:
    background: The game's background image.
    bird-wingup: An image of the bird with its wing pointing upward.
        Use this and bird-wingdown to create a flapping bird.
    bird-wingdown: An image of the bird with its wing pointing downward.
        Use this and bird-wingup to create a flapping bird.
    pipe-end: An image of a pipe's end piece (the slightly wider bit).
        Use this and pipe-body to make pipes.
    pipe-body: An image of a slice of a pipe's body.  Use this and
        pipe-body to make pipes.
    """

    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folder
        (./images/).  All images are converted before being returned to
        speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        file_name = os.path.join('.', 'birdimages', img_file_name)
        img = pygame.image.load(file_name)
        # converting all images before use speeds up blitting
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            # images for animating the flapping bird -- animated GIFs are
            # not supported in pygame
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png'),}

class Bird(pygame.sprite.Sprite):
    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.2
    Fail_SINk_SPEED = 0.6
    CLIMB_SPEED = 0.25
    CLIMB_DURATION = 333.3
    REGION = CLIMB_DURATION / 3  # when far from the pipe, the bird can fluctuate in a certain region,wgx
    NEAR_COLLIDE = 30  # when inside the pipe, near collide distance, this define another state
    NEAR_PIPE = 0  # at what distance does the bird near the pipe

    def __init__(self, x, y, msec_to_climb, images):
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, action,state,delta_frames=1):
        if self.msec_to_climb > 0 and action == 1:
            if state==4 or state==5 or state == 2 or state == 3:
                self.y -= (2*Bird.CLIMB_SPEED * (1000.0 * delta_frames / 60))
            else:
                self.y -= (Bird.CLIMB_SPEED * (1000.0 * delta_frames / 60))
        else:
            if state == 4 or state == 5 or state == 2 or state == 3:
                self.y += 2*Bird.SINK_SPEED * (1000.0 * delta_frames / 60)
            else:
                self.y += Bird.SINK_SPEED * (1000.0 * delta_frames / 60)

    #  if the bird fails, sink the bird till it hit the bottom
    def sink(self, delta_frames=1):
        self.y += Bird.Fail_SINk_SPEED * (1000.0 * delta_frames / 60)

    @property
    def image(self):
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        else:
            return self._img_wingdown

    @property
    def mask(self):
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)

class PipePair(pygame.sprite.Sprite):
    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 2000
    ADD_EVENT = pygame.USEREVENT + 1
    ROOM_HIGHT = 2 * Bird.HEIGHT + 2 * PIECE_HEIGHT

    def __init__(self, pipe_end_img, pipe_body_img):
        """Initialises a new random PipePair.

        The new PipePair will automatically be assigned an x attribute of
        float(WIN_WIDTH - 1).

        Arguments:
        pipe_end_img: The image to use to represent a pipe's end piece.
        pipe_body_img: The image to use to represent one horizontal slice
            of a pipe's body.
        """
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False
        self.isNewPipe = True

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()   # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -  # fill window from top to bottom
             3 * Bird.HEIGHT -  # make room for bird to fit through
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT  # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i * PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        self.center = (top_pipe_end_y + bottom_pipe_end_y) / 2  # center of pipe-room,wgx

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)
        self.top_y = top_pipe_end_y
        self.bottom_y = bottom_pipe_end_y

    @property
    def top_height_px(self):
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):
        """Update the PipePair's position.

        Attributes:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        self.x -= 0.18 * 1000.0 * delta_frames /60

    def collides_with(self, bird):
        """Get whether the bird collides with a pipe in this PipePair.

        Arguments:
        bird: The Bird which should be tested for collision with this
            PipePair.
        """
        return pygame.sprite.collide_mask(self, bird)

def chooseAct(Net,s,input,weight_trace_d1,weight_trace_d2):
    for i_train in range(500):
        out, dw = Net(input)
        # 更新权重
        # Net.UpdateWeight(10, dw[5][0])
        # Net.UpdateWeight(12, dw[5][1])
        # Net.UpdateWeight(11, dw[6][0])
        # rstdp
        weight_trace_d1 *= trace_decay
        weight_trace_d1 += dw[0][0]
        weight_trace_d2 *= trace_decay
        weight_trace_d2 += dw[1][0]
        if torch.max(out) > 0:
            return torch.argmax(out),weight_trace_d1,weight_trace_d2,Net

def judgeState(bird, pipes, collide):
    # bird's x and y coordinate in the left top of the image
    dist = bird.y + Bird.HEIGHT / 2 - WIN_HEIGHT / 2
    isNew = False
    index = -1
    state = -1
    if collide:
        state = 8
        return state
    for p in pipes:
        if p.x + PipePair.WIDTH - Bird.HEIGHT / 4 < bird.x and not p.score_counted:
            continue
        if p.x - Bird.NEAR_PIPE <= bird.x + Bird.HEIGHT and \
                p.x + PipePair.WIDTH - Bird.HEIGHT / 4 >= bird.x:

            p_top_y = p.top_y + PipePair.PIECE_HEIGHT
            p_bottom_y = p.bottom_y - PipePair.PIECE_HEIGHT
            if p.center - bird.y - Bird.HEIGHT / 2 >= 0 and bird.y >= p_top_y + Bird.NEAR_COLLIDE / 2:
                state = 0
            elif bird.y - p.center + Bird.HEIGHT / 2 > 0 and bird.y + Bird.HEIGHT <= p_bottom_y - Bird.NEAR_COLLIDE / 2:
                state = 1
            elif bird.y < p_top_y + Bird.NEAR_COLLIDE / 2 and bird.y > p_top_y - 10:
                state = 6
            elif bird.y + Bird.HEIGHT > p_bottom_y - Bird.NEAR_COLLIDE / 2 and bird.y + Bird.HEIGHT < p_bottom_y + 10:
                state = 7
            if state > -0.5:
                index = 1
        elif p.x > bird.x + Bird.HEIGHT + Bird.NEAR_PIPE:
            state = blankState(bird, p.center)
            if p.isNewPipe:
                isNew = True
            p.isNewPipe = False
            index = 1
        if index > 0:  # only judge the nearest and not passed pipe
            dist = bird.y + Bird.HEIGHT / 2 - p.center
            break
    if index < -0.5:  # no pipe left, key the bird in the middle
        pos = WIN_HEIGHT / 2
        dist = bird.y + Bird.HEIGHT / 2 - pos
        state = blankState(bird, pos)

    return state, dist, isNew

def blankState(bird, center):  # judge the state before passing the pipe
    realHeight = (PipePair.ROOM_HIGHT - Bird.HEIGHT) / 2
    if center - bird.y - Bird.HEIGHT / 2 >= 0 and \
            center - bird.y - Bird.HEIGHT / 2 < realHeight - Bird.NEAR_COLLIDE / 2:
        state = 0
    elif bird.y - center + Bird.HEIGHT / 2 >= 0 and \
            bird.y - center + Bird.HEIGHT / 2 < realHeight - Bird.NEAR_COLLIDE / 2:
        state = 1
    elif center - bird.y - Bird.HEIGHT / 2 >= realHeight - Bird.NEAR_COLLIDE / 2 and \
            center - bird.y - Bird.HEIGHT / 2 < realHeight - Bird.NEAR_COLLIDE / 2 + Bird.REGION:
        state = 2
    elif bird.y - center + Bird.HEIGHT / 2 >= realHeight - Bird.NEAR_COLLIDE / 2 and \
            bird.y - center + Bird.HEIGHT / 2 < realHeight - Bird.NEAR_COLLIDE / 2 + Bird.REGION:
        state = 3
    elif bird.y + Bird.HEIGHT / 2 <= center - (realHeight - Bird.NEAR_COLLIDE / 2 + Bird.REGION):
        state = 4
    elif bird.y + Bird.HEIGHT / 2 >= center + realHeight - Bird.NEAR_COLLIDE / 2 + Bird.REGION:
        state = 5
    return state

def getReward(state,lastState,smallerError,isNewPipe):
    if state == 0 or state == 1:
        reward = 6
    elif state == 2 or state == 3:
        if lastState == state and not isNewPipe:
            if smallerError:
                reward = 3
            else:
                reward = -5
        else:
            reward = -3
    elif state == 4 or state == 5:
        if lastState == state and not isNewPipe:
            if smallerError:
                reward = 3
            else:
                reward = -8
        else:
            reward = -5
    elif state == 6 or state == 7:
        if lastState == state and not isNewPipe:
            if smallerError:
                reward = 3
            else:
                reward = -3
        else:
            reward = -3
    elif state == 8:   #  collide
        reward = -100
    return reward

def updateNet(Net,reward, action, state,weight_trace_d1,weight_trace_d2):
    r = torch.ones((num_state, num_state * num_action), dtype=torch.float)
    r[state, state * num_action + action] = reward
    dw_d1 = r * weight_trace_d1
    dw_d2 = -1 * r * weight_trace_d2
    Net.UpdateWeight(0, state, num_action, dw_d1)
    Net.UpdateWeight(1, state, num_action, dw_d2)
    return Net

if __name__=="__main__":
    #定义网络
    num_state=9
    num_action=2
    weight_exc=1
    weight_inh=-0.5
    trace_decay = 0.8
    DM = BDMSNN(num_state, num_action, weight_exc, weight_inh, "lif")
    con_matrix1 = torch.zeros((num_state, num_state * num_action), dtype=torch.float)
    for i in range(num_state):
        for j in range(num_action):
            con_matrix1[i, i * num_action + j] = weight_exc
    weight_trace_d1 = torch.zeros(con_matrix1.shape, dtype=torch.float)
    weight_trace_d2 = torch.zeros(con_matrix1.shape, dtype=torch.float)

    #定义游戏场景
    pygame.init()
    WIN_HEIGHT = 512
    WIN_WIDTH = 284 * 2  # image size: 284x512 px; tiled twice
    heighest = 0
    iteration=0
    contTime = 0  # number of times to restart
    display_frame=0
    while iteration < 20:       #  restart the game for reinforcement learning, wgx
        display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')
        images = load_images()
        bird = Bird(250, int(WIN_HEIGHT / 2 - Bird.HEIGHT / 2), 2,
                    (images['bird-wingup'], images['bird-wingdown']))

        clock = pygame.time.Clock()
        score_font = pygame.font.SysFont(None, 25, bold=True)  # default font
        info_font = pygame.font.SysFont(None, 50, bold=True)
        collide = paused = False
        frame_clock = 0
        pipes = deque()
        score = 0
        lastDist = 0
        lastState = 0 #init
        state = lastState
        while not collide:
            # 输入
            input = torch.zeros((num_state), dtype=torch.float)
            clock.tick(60)
            if frame_clock %2==0 or frame_clock==1:
                state, dist, isNewPipe = judgeState(bird, pipes, collide)  # judge the bird's state
                lastState = state
                lastDist = dist
                input[state]=2
                action,weight_trace_d1,weight_trace_d2,DM = chooseAct(DM,state,input,weight_trace_d1,weight_trace_d2)
                print("state, dist:", state, dist)
                print("state, action:",state,action)
            if not (paused or frame_clock % (60 * PipePair.ADD_INTERVAL / 1000.0)):
                pygame.event.post(pygame.event.Event(PipePair.ADD_EVENT))

            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                    collide = True
                elif e.type == KEYUP and e.key in (K_PAUSE, K_p):
                    paused = not paused
                elif e.type == PipePair.ADD_EVENT:
                    pp = PipePair(images['pipe-end'], images['pipe-body'])
                    pipes.append(pp)
            if paused:
                continue  # don't draw anything
            # check for collisions
            pipe_collision = any(p.collides_with(bird) for p in pipes)
            if pipe_collision or 0 >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                collide = True
            for x in (0, WIN_WIDTH / 2):
                display_surface.blit(images['background'], (x, 0))
            while pipes and not pipes[0].visible:
                pipes.popleft()
            for p in pipes:
                p.update()
                display_surface.blit(p.image, p.rect)
            bird.update(action,state)
            display_surface.blit(bird.image, bird.rect)
            if frame_clock %2==0 or frame_clock==1 or collide:
                # judge the state and update the value function
                dist = 0
                if collide:
                    nextState = 8
                    isNewPipe = False
                else:
                    nextState, dist, isNewPipe = judgeState(bird, pipes, collide)  # judge the bird's state
                    print("next state:", nextState)
                print("lastdist, dist:", lastDist,dist)
                isSmallerError = False
                if state == nextState:
                    isSmallerError = False
                    if lastDist <= 0:
                        if lastDist < dist:
                            isSmallerError = True
                    else:
                        if lastDist > dist:
                            isSmallerError = True
                if frame_clock>0 and not collide:
                    reward = getReward(nextState, state, isSmallerError, isNewPipe)
                    print("reward:", reward)
                    DM=updateNet(DM,reward, action, state,weight_trace_d1,weight_trace_d2)
                state = nextState  #going on the next state
                weight_trace_d1 = torch.zeros(con_matrix1.shape, dtype=torch.float)
                weight_trace_d2 = torch.zeros(con_matrix1.shape, dtype=torch.float)
                DM.reset()
                display_frame += 1
                # update and display score
            for p in pipes:
                if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                    score += 1
                    p.score_counted = True

            score_surface = score_font.render('Current score: ' + str(score), True, (0, 0, 0))  # current score
            score_x = WIN_WIDTH / 2 - 3 * score_surface.get_width() / 4
            display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))
            if heighest < score:
                heighest = score
            score_surface_h = score_font.render('Highest score: ' + str(heighest), True,
                                                (0, 0, 0))  # heighest score
            score_x_h = 4 * WIN_WIDTH / 5 - 1.2 * score_surface.get_width() / 3
            display_surface.blit(score_surface_h, (score_x_h, PipePair.PIECE_HEIGHT))
            score_surface_i = score_font.render('Attempts: ' + str(iteration), True, (0, 0, 0))  # heighest score
            score_x_i = 10
            display_surface.blit(score_surface_i, (score_x_i, PipePair.PIECE_HEIGHT))
            frame_clock += 1
            pygame.display.flip()

        #  if collide, display the fail information, for 2 frames
        cct = 0
        while (bird.y < WIN_HEIGHT - Bird.HEIGHT - 3):
            clock.tick(60)
            for x in (0, WIN_WIDTH / 2):
                display_surface.blit(images['background'], (x, 0))
            while pipes and not pipes[0].visible:
                pipes.popleft()
            for p in pipes:
                display_surface.blit(p.image, p.rect)
            if cct >= 6:
                bird.sink()
            display_surface.blit(bird.image, bird.rect)
            fail_infor = info_font.render('Game over !', True, (255, 60, 30))  # current score
            pos_x = WIN_WIDTH / 2 - fail_infor.get_width() / 2
            pos_y = WIN_HEIGHT / 2 - 100
            display_surface.blit(fail_infor, (pos_x, pos_y))
            #  display the score
            score_surface = score_font.render('Current score: ' + str(score), True, (0, 0, 0))  # current score
            score_x = WIN_WIDTH / 2 - 3 * score_surface.get_width() / 4
            display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))
            if heighest < score:
                heighest = score
            score_surface_h = score_font.render('Highest score: ' + str(heighest), True,
                                                (0, 0, 0))  # heighest score
            score_x_h = 4 * WIN_WIDTH / 5 - 1.2 * score_surface.get_width() / 3
            display_surface.blit(score_surface_h, (score_x_h, PipePair.PIECE_HEIGHT))
            score_surface_i = score_font.render('Attempts: ' + str(iteration), True, (0, 0, 0))  # heighest score
            score_x_i = 10
            display_surface.blit(score_surface_i, (score_x_i, PipePair.PIECE_HEIGHT))
            pygame.display.flip()
            cct += 1
        if heighest < score:
            heighest = score
        contTime += 1
        iteration += 1