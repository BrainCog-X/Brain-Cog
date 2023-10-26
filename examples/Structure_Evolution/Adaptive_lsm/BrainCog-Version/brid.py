import torch, os
import pygame
from pygame.locals import *
from collections import deque
from random import randint
import numpy as np
import nsganet as engine
from pymop.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation
from lsmmodel import SNN
import torch.nn.functional as F

from tools.update_weights import stdp,bcm
import matplotlib.pyplot as plt
os.environ["SDL_VIDEODRIVER"] = "dummy"
steps=30000
seeds=50
result=np.zeros([seeds,int(steps/2)])
t=[i for i in range(steps)]



def randbool(size, p):
    return torch.rand(*size) < p

class Evolve(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int64)
        self.xl = lb
        self.xu = ub
        self._n_evaluated = 0  # keep track of how many architectures are sampled


    def _evaluate(self, x, out, *args, **kwargs):
        
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('Network= {}'.format(arch_id))
            objs[i, 0] = np.linalg.matrix_rank(x[i])
            objs[i, 1] = 0
            self._n_evaluated += 1
        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")
    
    # report generation info to files
    print("generation = {}".format(gen))
    print("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    print('Best Genome id= {}'.format(np.argmin(pop_obj[:, 0])))

def load_images():
    """
    Flappy Bird中load图像
    :return:load的图像
    """

    def load_image(img_file_name):
        file_name = os.path.join('/home/panwenxuan/raw', 'birdimages', img_file_name)
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
            'bird-wingdown': load_image('bird_wing_down.png'), }


class Bird(pygame.sprite.Sprite):
    """
    Flappy Bird类
    """
    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.2
    Fail_SINk_SPEED = 0.6
    CLIMB_SPEED = 0.25
    CLIMB_DURATION = 333.3
    REGION = CLIMB_DURATION / 3
    NEAR_COLLIDE = 30
    NEAR_PIPE = 0

    def __init__(self, x, y, msec_to_climb, images):
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, action, state, delta_frames=1):
        """
        更新小鸟的位置
        :param action: 输入行为
        :param state:输入状态
        :param delta_frames:Fault
        :return:None
        """
        if self.msec_to_climb > 0 and action == 1:
            if state == 4 or state == 5 or state == 2 or state == 3:
                self.y -= (2 * Bird.CLIMB_SPEED * (1000.0 * delta_frames / 60))
            else:
                self.y -= (Bird.CLIMB_SPEED * (1000.0 * delta_frames / 60))
        else:
            if state == 4 or state == 5 or state == 2 or state == 3:
                self.y += 2 * Bird.SINK_SPEED * (1000.0 * delta_frames / 60)
            else:
                self.y += Bird.SINK_SPEED * (1000.0 * delta_frames / 60)

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
    """
    Flappy Bird 中的管子类
    """
    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 2000
    ADD_EVENT = pygame.USEREVENT + 1
    ROOM_HIGHT = 2 * Bird.HEIGHT + 2 * PIECE_HEIGHT

    def __init__(self, pipe_end_img, pipe_body_img):
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False
        self.isNewPipe = True

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()  # speeds up blitting
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

        self.center = (top_pipe_end_y + bottom_pipe_end_y) / 2

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)
        self.top_y = top_pipe_end_y
        self.bottom_y = bottom_pipe_end_y

    @property
    def top_height_px(self):
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):
        self.x -= 0.18 * 1000.0 * delta_frames / 60

    def collides_with(self, bird):
        return pygame.sprite.collide_mask(self, bird)


def judgeState(bird, pipes, collide):
    """
    根据小鸟和管子之间的位置关系判断当前状态
    :param bird:传入小鸟的各项属性
    :param pipes:传入管子的各项属性
    :param collide:是否发生碰撞
    :return:状态，距离，是否是新的管子
    """
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


def blankState(bird, center):
    """
    judgeState中调用的判断状态的函数 根据鸟的位置和管子中心的距离来判断
    :param bird: 传入小鸟的各项属性
    :param center:中心
    :return:状态
    """
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


def getReward(state, lastState, smallerError, isNewPipe):
    """
    根据状态和距离的变化获得奖励
    :param state: 执行行为后的当前状态
    :param lastState:执行行为之前的上一状态
    :param smallerError:距离是否变小
    :param isNewPipe:是否是新的管子
    :return:奖励
    """
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
    elif state == 8:  # collide
        reward = -100
    return reward




if __name__ == "__main__":

    n_agent=1
    num = 8
    p_amount = int(num * num / 10)
    s_amount = 4
    num_state = 9
    num_action = 2
    weight_exc = 1
    weight_inh = -0.5
    trace_decay = 0.8
    gens=1000
    for seed in range(seeds):

        kkk = Evolve(n_var=num*num, 
                        n_obj=2, n_constr=0)
        method = engine.nsganet(pop_size=n_agent,
                                sampling=RandomSampling(var_type='custom'),
                                mutation=BinaryBitflipMutation(),
                                n_offsprings=10,
                                eliminate_duplicates=True)
        kres=minimize(kkk,
                        method,
                        callback=do_every_generations,
                        termination=('n_gen', gens))
    
        pop_var = kres.X
        pop_obj = kres.F        
        lm=torch.from_numpy(pop_var[np.argmin(pop_obj[:, 0])].reshape(num,num))
    
        model = SNN(ins=9,num_classes=2,n_agent=n_agent,device='cuda:0',liquid_size=num,lsm_tau=2,lsm_th=0.2,connectivity_matrix=lm.to('cuda:0').float())
        model.to('cuda:0')

        con_matrix1 = torch.zeros((num_state, num_state * num_action), dtype=torch.float)
        for i in range(num_state):
            for j in range(num_action):
                con_matrix1[i, i * num_action + j] = weight_exc
        weight_trace_d1 = torch.zeros(con_matrix1.shape, dtype=torch.float)
        weight_trace_d2 = torch.zeros(con_matrix1.shape, dtype=torch.float)

        pygame.init()
        WIN_HEIGHT = 512
        WIN_WIDTH = 284 * 2
        heighest = 0
        contTime = 0
        display_frame = 0
        display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')
        images = load_images()
        bird = Bird(250, int(WIN_HEIGHT / 2 - Bird.HEIGHT / 2), 2,
                    (images['bird-wingup'], images['bird-wingdown']))

        clock = pygame.time.Clock()
        score_font = pygame.font.SysFont(None, 25, bold=True)
        info_font = pygame.font.SysFont(None, 50, bold=True)
        collide = paused = False
        frame_clock = 0
        pipes = deque()
        score = 0
        lastDist = 0
        lastState = 0  # init
        state = lastState
        i = 0
        num_reward = []
        num_score = []
        reward=1
        while not collide:
            i = i + 1
            if i > steps:
                break
            clock.tick(60)
            if frame_clock % 2 == 0 or frame_clock == 1:
                state, dist, isNewPipe = judgeState(bird, pipes, collide)
                lastState = state
                lastDist = dist
                action= model(F.one_hot(torch.tensor([state]), num_classes=num_state).to('cuda:0').float()).cpu().detach().numpy()
                action=int(np.argmax(action,axis=1))
                print(i)

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
            bird.update(action, state)
            display_surface.blit(bird.image, bird.rect)
            if frame_clock % 2 == 0 or frame_clock == 1 or collide:
                dist = 0
                if collide:
                    nextState = 8
                    isNewPipe = False
                else:
                    nextState, dist, isNewPipe = judgeState(bird, pipes, collide)  # judge the bird's state
                    print("next state:", nextState)
                print("lastdist, dist:", lastDist, dist)
                isSmallerError = False
                if state == nextState:
                    isSmallerError = False
                    if lastDist <= 0:
                        if lastDist < dist:
                            isSmallerError = True
                    else:
                        if lastDist > dist:
                            isSmallerError = True
                if frame_clock > 0 and not collide:
                    reward = getReward(nextState, state, isSmallerError, isNewPipe)
                    print("reward:", reward)
                    num_reward.append(reward)
                    bcmreward=np.array([reward, reward])
                    # bcm(model,bcmreward, input=input)
                state = nextState  # going on the next state
                weight_trace_d1 = torch.zeros(con_matrix1.shape, dtype=torch.float)
                weight_trace_d2 = torch.zeros(con_matrix1.shape, dtype=torch.float)
                model.reset()
                display_frame += 1
            for p in pipes:
                if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                    score += 1
                    p.score_counted = True
            num_score.append(score)
            score_surface = score_font.render('Current score: ' + str(score), True, (0, 0, 0))  # current score
            score_x = WIN_WIDTH / 2 - 3 * score_surface.get_width() / 4
            display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))
            if heighest < score:
                heighest = score
            score_surface_h = score_font.render('Highest score: ' + str(heighest), True,
                                                (0, 0, 0))  # heighest score
            score_x_h = 4 * WIN_WIDTH / 5 - 1.2 * score_surface.get_width() / 3
            display_surface.blit(score_surface_h, (score_x_h, PipePair.PIECE_HEIGHT))
            score_surface_i = score_font.render('Attempts: 0', True, (0, 0, 0))  # heighest score
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
            score_surface_i = score_font.render('Attempts: 0', True, (0, 0, 0))  # heighest score
            score_x_i = 10
            display_surface.blit(score_surface_i, (score_x_i, PipePair.PIECE_HEIGHT))
            pygame.display.flip()
            cct += 1
        if heighest < score:
            heighest = score
        contTime += 1
        num_reward_np = np.array(num_reward)
        print(num_reward_np)
        k=num_reward_np.shape[0]
        result[seed,:k]=num_reward_np

