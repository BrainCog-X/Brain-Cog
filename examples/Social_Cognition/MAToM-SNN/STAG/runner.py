import os
import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing as mp

from common_sr.srollout import RolloutWorker
from agents.sagent import Agents
from common_sr.replay_buffer import ReplayBuffer

import time
from tqdm import tqdm

class Runner:
    def __init__(self, env, args):
        self.env = env

        self.agents = Agents(args)

        self.rolloutWorker = RolloutWorker(env, self.agents, args)

        if not args.evaluate:
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + args.exp_dir
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, -1
        pbar = tqdm(self.args.n_steps)
        if self.args.load_model == False:
            while time_steps < self.args.n_steps:
                # print('Run {}, time_steps {}'.format(num, time_steps))
                if time_steps // self.args.evaluate_cycle > evaluate_steps:
                    win_rate, episode_reward = self.evaluate()
                    # episode_reward = [i for i in [2, 3]]
                    self.episode_rewards.append(episode_reward)
                    # self.plt(time_steps // self.args.evaluate_cycle)
                    # print(time_steps // self.args.evaluate_cycle)
                    evaluate_steps += self.args.evaluate_epoch
                # 收集self.args.n_episodes个episodes
                episodes = []
                start = time.time()
                episode_batch, _, _, steps = self.rolloutWorker.generate_episode()
                end = time.time()
                # print(end - start, 'sample with multiprocessing:', self.args.process)
                time_steps += steps
                pbar.update(steps)
                self.buffer.store_episode(episode_batch)

                start = time.time()
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    if self.args.alg.find('o') > -1:
                        self.agents.train(mini_batch, train_steps, self.args.epsilon)
                    else:
                        self.agents.train(mini_batch, train_steps)
                    train_steps += 1
                end = time.time()
                    # print(end - start, 'training')
        pbar.close()
        win_rate, episode_reward = self.evaluate()
        # print('win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        if self.args.load_model == False:
            self.plt(num)

    def evaluate(self):
        win_number = 0
        episode_rewards = (0, 0)  # cumulative rewards

        _, episode_rewards, win_tag, _ = self.rolloutWorker.generate_episode(evaluate=True)

        episode_rewards = [episode_rewards[i] / self.args.evaluate_epoch / self.args.process for i in range(len(episode_rewards))]
        return win_number / self.args.evaluate_epoch, episode_rewards

    def plt(self, num):
        # plt.figure()
        # plt.ylim([0, 105])
        # plt.cla()
        # plt.subplot(2, 1, 1)
        # plt.plot(range(len(self.win_rates)), self.win_rates)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('win_rates')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('episode_rewards')
        #
        # plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        # np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        # np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)
        # plt.close()
        # plt.figure()
        # plt.ylim([0, 105])
        # plt.cla()
        # plt.plot(2, 1, 1)
        # plt.plot(range(len(self.episode_rewards)), self.episode_rewards[0][0])
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('episode_rewards_A')
        #
        # plt.plot(2, 1, 2)
        # plt.plot(range(len(self.episode_rewards)), self.episode_rewards[0][1])
        # plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        # plt.ylabel('episode_rewards_B')
        # plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        # np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)   #
        # print(self.episode_rewards)
        # plt.close()