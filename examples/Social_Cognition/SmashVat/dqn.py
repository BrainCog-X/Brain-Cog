import os
import time
import random
from itertools import count
from collections import namedtuple, deque

import numpy as np
import pandas as pd
import torch
import imageio
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from side_effect_eval import *
from qnets import *
from environment import *

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch


class AnseEmpDQN:
    def __init__(self, env, net_type='SNN',
                 init_buffer_size=10000, replay_buffer_size=100000, batch_size=100, target_update_interval=1000,
                 weight_sep=20, weight_emp_impact=None):

        self.env = env

        input_dim = 3
        output_dim = env.action_space.n
        assert net_type in ['SNN', 'ANN']
        self.net_type = net_type
        if self.net_type == 'SNN':
            self.policy_net = SNNQnet(input_dim, output_dim).cuda()
            self.target_net = SNNQnet(input_dim, output_dim).cuda()
        elif self.net_type == 'ANN':
            self.policy_net = CNNQnet(input_dim, output_dim).cuda()
            self.target_net = CNNQnet(input_dim, output_dim).cuda()

        self.init_buffer_size = init_buffer_size
        self.replay_buffer_size = replay_buffer_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        # empathy
        self.num_others = env.num_humans
        self.baseline = StepwiseInactionModel(noop_action=env.actions.noop)
        self.baselines_others = [StepwiseInactionModel(noop_action=env.actions.noop)] * self.num_others
        self.deviation = AttainableUtilityMeasure(uf_num=30, uf_discount=0.99)
        self.weight_sep = weight_sep
        self.weight_emp_impact = weight_emp_impact if weight_emp_impact is not None else weight_sep

    def state2tensor(self, state):
        state_arr = self.env._decode(state)
        array = np.zeros([3] + list(self.env.p.map_shape))
        array[0] = self.env.map  # [0]为环境信息
        for i, pos in enumerate(self.env.p.vat_pos):
            array[0][pos] = self.env.cells.vat if state_arr[i] else self.env.cells.empty  # 根据state还原各个缸的状态，因为replay_buffer中存的状态对应的map与当前的可能不一样
        agent_pos = tuple(state_arr[self.env.num_vats:self.env.num_vats + 2])
        array[1][agent_pos] = 1  # [1]为agent位置信息
        for pos in self.env.human_pos:
            array[2][tuple(pos)] = 1  # [2]为human位置信息
        return torch.Tensor(array).float().cuda()

    def epsilon_greedy(self, net, state, epsilon):
        num_actions = self.env.action_space.n
        p = np.ones(num_actions) * epsilon / num_actions
        state_tensor = self.state2tensor(state).unsqueeze(0)
        best_action = torch.argmax(net(state_tensor)).item()
        p[best_action] += 1 - epsilon
        action = np.random.choice(self.env.action_space.n, p=p)
        return action

    def train(self, lr=1e-3, num_episodes=10000, gamma=0.99,
              epsilon_start=1, epsilon_end=0.01, decay_start=0.05, decay_end=0.95,
              checkpoint_interval=1000,
              checkpoint_dir='./models/',
              log_dir='./log'):
        policy_net_opt = optim.Adam(self.policy_net.parameters(), lr=lr)

        epsilons = [epsilon_end] * num_episodes
        decay_start_episode = int(num_episodes * decay_start)
        decay_end_episode = int(num_episodes * decay_end)
        epsilons[0:decay_start_episode] = np.full(decay_start_episode, epsilon_start)
        epsilons[decay_start_episode:decay_end_episode] = np.linspace(epsilon_start, epsilon_end, decay_end_episode - decay_start_episode)

        tb_logger = SummaryWriter(log_dir=log_dir)
        pd_timestep_logger = pd.DataFrame(columns=['loss', 'reward', 'impact', 'aup_impact', 'empathy_impact'])
        pd_episode_logger = pd.DataFrame(columns=['step', 'ep_reward', 'ep_reward_mean',
                                                  'ep_impact', 'ep_aup_impact', 'ep_empathy_impact',
                                                  'num_vat_broken', 'num_human_saved'])

        def optimize_net():
            transitions = self.replay_buffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            state_batch = torch.stack([self.state2tensor(state[0]) for state in batch.state]).float().cuda()
            action_batch = torch.tensor(batch.action).unsqueeze(-1).cuda()
            reward_batch = torch.tensor(batch.reward, dtype=torch.float).cuda()
            next_state_batch = torch.stack([self.state2tensor(state[0]) for state in batch.next_state]).float().cuda()
            done_batch = torch.tensor(batch.done).cuda()
            best_actions = self.policy_net(next_state_batch).max(1)[1].detach()
            next_state_values = self.target_net(next_state_batch).gather(1, best_actions.unsqueeze(1)).squeeze(1).detach()

            expected_state_action_values = reward_batch + gamma * next_state_values * torch.logical_not(done_batch)
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            policy_net_opt.zero_grad()
            loss.backward()
            policy_net_opt.step()
            tb_logger.add_scalar('loss', loss.item(), total_step)
            pd_timestep_logger.loc[total_step, 'loss'] = loss.item()

        def calculate_impact(prev_states, prev_action, current_states):
            if self.weight_sep==0 and self.weight_emp_impact==0:
                return 0, 0, 0

            prev_state_agent = prev_states[0]
            current_state_agent = current_states[0]
            prev_states_others = prev_states[1:]
            current_states_others = current_states[1:]

            baseline_state_agent = self.baseline.calculate(prev_state_agent, prev_action, current_state_agent)
            self.deviation.update(prev_state_agent, prev_action, current_state_agent)
            dev_self = self.deviation.calculate(current_state_agent, baseline_state_agent, lambda x: abs(np.minimum(0, x)))
            weighted_dev_self = -self.weight_sep * dev_self

            dev_others = []
            for prev_state, current_state, baseline in zip(prev_states_others, current_states_others, self.baselines_others):
                baseline_state = baseline.calculate(prev_state, prev_action, current_state)
                dev_others.append(self.deviation.calculate(current_state, baseline_state, lambda x: x))
            dev_others_mean = sum(dev_others) / len(dev_others) if len(dev_others) > 0 else 0
            weighted_dev_others = self.weight_emp_impact * dev_others_mean
            total_impact = weighted_dev_self + weighted_dev_others
            return total_impact, weighted_dev_self, weighted_dev_others

        # 初始化replay buffer
        state = self.env.reset()
        for i in range(self.init_buffer_size):
            # action = self.epsilon_greedy(self.empathy_net, state[0], epsilon_start)
            action = self.epsilon_greedy(self.policy_net, state[0], epsilon_start)
            next_state, reward, done, info = self.env.step(action)

            if self.weight_sep != 0 or self.weight_emp_impact != 0:  # 如果都等于0则退化为标准DQN
                impact, _, _ = calculate_impact(state, action, next_state)
                reward += impact
                reward /= (self.weight_sep + self.weight_emp_impact) / 2  # 正则化操作，防止reward绝对值过大使得网络发散

            self.replay_buffer.push(state, action, reward, next_state, done)
            if done:
                state = self.env.reset()
            else:
                state = next_state

        # 开始训练
        total_step = 0
        for episode in range(num_episodes):
            if episode % checkpoint_interval == 0 and episode != 0:
                torch.save(self.policy_net.state_dict(), checkpoint_dir + f"/checkpoint_{episode}.pth")

            tb_logger.add_scalar('epsilon', epsilons[episode], episode + 1)
            state = self.env.reset()
            if episode % 100 == 0:
                print('Episode {} of {}'.format(episode + 1, num_episodes))

            episode_reward = 0
            episode_impact = 0
            ep_aup_impact = 0
            ep_empathy_impact = 0
            for step in count():
                if total_step % self.target_update_interval == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                action = self.epsilon_greedy(self.policy_net, state[0], epsilons[episode])
                next_state, reward, done, info = self.env.step(action)

                impact, aup_impact, empathy_impact = calculate_impact(state, action, next_state)
                if self.weight_sep != 0 or self.weight_emp_impact != 0:  # 如果都等于0则退化为标准DQN
                    reward += impact
                    reward /= (self.weight_sep + self.weight_emp_impact) / 2

                episode_reward += reward
                episode_impact += impact
                ep_aup_impact += aup_impact
                ep_empathy_impact += empathy_impact

                self.replay_buffer.push(state, action, reward, next_state, done)
                optimize_net()

                pd_timestep_logger.loc[total_step, 'reward'] = reward
                pd_timestep_logger.loc[total_step, 'impact'] = impact
                pd_timestep_logger.loc[total_step, 'aup_impact'] = aup_impact
                pd_timestep_logger.loc[total_step, 'empathy_impact'] = empathy_impact

                if done:
                    # print(f'step: {step}, reward: {episode_reward:.2f}')
                    tb_logger.add_scalar('step', step + 1, episode + 1)
                    tb_logger.add_scalar('reward', episode_reward, episode + 1)
                    tb_logger.add_scalar('ep-reward-mean', episode_reward / (step + 1), episode + 1)
                    tb_logger.add_scalar('impact', episode_impact, episode + 1)

                    pd_episode_logger.loc[episode, 'step'] = step + 1
                    pd_episode_logger.loc[episode, 'ep_reward'] = episode_reward
                    pd_episode_logger.loc[episode, 'ep_reward_mean'] = episode_reward / (step + 1)
                    pd_episode_logger.loc[episode, 'ep_impact'] = episode_impact
                    pd_episode_logger.loc[episode, 'ep_aup_impact'] = ep_aup_impact
                    pd_episode_logger.loc[episode, 'ep_empathy_impact'] = ep_empathy_impact

                    env_map = self.env.map
                    num_vat_broken = 0
                    num_human_saved = 0
                    for pos in self.env.p.vat_pos:
                        if env_map[pos] != self.env.cells.vat:
                            num_vat_broken += 1
                            if pos in self.env.p.human_pos:
                                num_human_saved += 1
                    pd_episode_logger.loc[episode, 'num_vat_broken'] = num_vat_broken
                    pd_episode_logger.loc[episode, 'num_human_saved'] = num_human_saved
                    break
                else:
                    state = next_state
                total_step += 1
        tb_logger.close()
        pd_timestep_logger.to_csv(log_dir + '/timestep_logger.csv', index=False)
        pd_episode_logger.to_csv(log_dir + '/episode_logger.csv', index=False)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy_net.state_dict(), path + "/policy_net.pth")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + "/policy_net.pth"))

    def run(self, gif_name=None):
        self.policy_net.eval()
        obs = self.env.reset()
        images = []
        for step in count():
            # self.env.render()
            images.append(self.env.render(mode='rgb_array'))

            obs_tensor = self.state2tensor(obs[0]).unsqueeze(0)
            action_p = self.policy_net(obs_tensor)
            action = torch.argmax(self.policy_net(obs_tensor)).item()
            print(self.env.actions(action))
            next_state, reward, done, _ = self.env.step(action)

            time.sleep(1)
            if done:
                images.append(self.env.render(mode='rgb_array'))
                images.append(self.env.render(mode='rgb_array'))
                break
            else:
                obs = next_state
        if gif_name is not None:
            imageio.mimsave(gif_name, images, 'GIF', duration=0.5)


def set_seed(seed=114514):
    random.seed(seed)  # replay buffer 中使用了random.sample
    np.random.seed(seed)  # e-greedy 中使用了np.random_choice
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    set_seed(1919810)
    env = BasicVatGoalEnv()
    env.render()
    model = AnseEmpDQN(env, net_type='ANN', init_buffer_size=10000, replay_buffer_size=100000, batch_size=100, target_update_interval=1000)

    model.train(lr=1e-3, num_episodes=10000, gamma=0.99,
                epsilon_start=1, epsilon_end=0.01, decay_start=0.05, decay_end=0.95,
                checkpoint_interval=1000,
                checkpoint_dir='./models/ANN-BasicVatGoalEnv-test',
                log_dir='./log/ANN-BasicVatGoalEnv-test')
    model.save("./models/ANN-BasicVatGoalEnv-test")
    model.load("./models/ANN-BasicVatGoalEnv-test")
    model.run("ANN-BasicVatGoalEnv-test.gif")


if __name__ == '__main__':
    main()
