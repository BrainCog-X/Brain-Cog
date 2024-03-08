import numpy as np
import torch

from environments import Env


# 由ToCMLearner的函数创建
class ToCMMemory:
    def __init__(self, capacity, sequence_length, action_size, obs_size, n_agents, device, env_type):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.action_size = action_size
        self.obs_size = obs_size
        self.device = device  # 加入了device
        self.env_type = env_type
        self.init_buffer(n_agents, env_type)  # TODO

    def init_buffer(self, n_agents, env_type):  # 初始对环境进行采样，观察和动作都是np.array  # TODO init_buffer specially?
        self.observations = np.empty((self.capacity, n_agents, self.obs_size), dtype=np.float32)
        self.actions = np.empty((self.capacity, n_agents, self.action_size), dtype=np.float32)
        self.av_actions = np.empty((self.capacity, n_agents, self.action_size),  # 3, 5
                                   dtype=np.float32) if env_type == Env.STARCRAFT or env_type == Env.MPE else None  # TODO need mask?
        self.rewards = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.fake = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.last = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.next_idx = 0
        self.n_agents = n_agents
        self.full = False

    def append(self, obs, action, reward, done, fake, last, av_action):
        if self.actions.shape[-2] != action.shape[-2]:
            self.init_buffer(action.shape[-2], self.env_type)
        for i in range(len(obs)):
            self.observations[self.next_idx] = obs[i]
            self.actions[self.next_idx] = action[i]
            if av_action is not None:
                self.av_actions[self.next_idx] = av_action[i]
            self.rewards[self.next_idx] = reward[i]
            self.dones[self.next_idx] = done[i]
            self.fake[self.next_idx] = fake[i]
            self.last[self.next_idx] = last[i]
            self.next_idx = (self.next_idx + 1) % self.capacity
            self.full = self.full or self.next_idx == 0

    def tenzorify(self, nparray):
        return torch.from_numpy(nparray).float()

    def sample(self, batch_size):
        return self.get_transitions(self.sample_positions(batch_size))

    def process_batch(self, val, idxs, batch_size):  # 这里全部传到了cuda上
        return torch.as_tensor(val[idxs].reshape(self.sequence_length, batch_size, self.n_agents, -1)).to(self.device)

    def get_transitions(self, idxs):
        batch_size = len(idxs)
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.process_batch(self.observations, vec_idxs, batch_size)[1:]
        reward = self.process_batch(self.rewards, vec_idxs, batch_size)[:-1]
        action = self.process_batch(self.actions, vec_idxs, batch_size)[:-1]
        av_action = self.process_batch(self.av_actions, vec_idxs, batch_size)[1:] if self.env_type == Env.STARCRAFT else None
        done = self.process_batch(self.dones, vec_idxs, batch_size)[:-1]
        fake = self.process_batch(self.fake, vec_idxs, batch_size)[1:]
        last = self.process_batch(self.last, vec_idxs, batch_size)[1:]

        return {'observation': observation, 'reward': reward, 'action': action, 'done': done, 
                'fake': fake, 'last': last, 'av_action': av_action}

    def sample_position(self):
        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.capacity if self.full else self.next_idx - self.sequence_length)
            idxs = np.arange(idx, idx + self.sequence_length) % self.capacity
            valid_idx = self.next_idx not in idxs[1:]  # Make sure data does not cross the memory index
        return idxs

    def sample_positions(self, batch_size):
        return np.asarray([self.sample_position() for _ in range(batch_size)])

    def __len__(self):
        return self.capacity if self.full else self.next_idx

    def clean(self):
        self.memory = list()
        self.position = 0
