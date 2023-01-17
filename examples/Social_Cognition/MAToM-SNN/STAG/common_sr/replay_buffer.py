import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        # self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape
        self.size = self.args.buffer_size
        self.episode_limit = self.args.episode_limit
        # memory management
        self.current_idx = 0
        self.current_size = 0
        # create the buffer to store info
        self.buffers = {'O': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        'U': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        # 's': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'R': np.empty([self.size, self.episode_limit, self.n_agents, 1]),
                        'O_NEXT': np.empty([self.size, self.episode_limit, self.n_agents, self.obs_shape]),
                        # 's_next': np.empty([self.size, self.episode_limit, self.state_shape]),
                        'AVAIL_U': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'AVAIL_U_NEXT': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'U_ONEHOT': np.empty([self.size, self.episode_limit, self.n_agents, self.n_actions]),
                        'PADDED': np.empty([self.size, self.episode_limit, 1]),
                        'TERMINATE': np.empty([self.size, self.episode_limit, 1])
                        }
        # thread lock
        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        batch_size = episode_batch['O'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['O'][idxs] = episode_batch['O']
            self.buffers['U'][idxs] = episode_batch['U']
            # self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['R'][idxs] = episode_batch['R']
            self.buffers['O_NEXT'][idxs] = episode_batch['O_NEXT']
            # self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['AVAIL_U'][idxs] = episode_batch['AVAIL_U']
            self.buffers['AVAIL_U_NEXT'][idxs] = episode_batch['AVAIL_U_NEXT']
            self.buffers['U_ONEHOT'][idxs] = episode_batch['U_ONEHOT']
            self.buffers['PADDED'][idxs] = episode_batch['PADDED']
            self.buffers['TERMINATE'][idxs] = episode_batch['TERMINATE']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_idx + inc <= self.size:
            idx = np.arange(self.current_idx, self.current_idx + inc)
            self.current_idx += inc
        elif self.current_idx < self.size:
            overflow = inc - (self.size - self.current_idx)
            idx_a = np.arange(self.current_idx, self.size)
            idx_b = np.arange(0, overflow)
            idx = np.concatenate([idx_a, idx_b])
            self.current_idx = overflow
        else:
            idx = np.arange(0, inc)
            self.current_idx = inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
