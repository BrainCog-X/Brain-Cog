import pickle

import numpy as np

from tools.Tools import get_data_path


class ExperimentEnvGlobalNetworkSurvival:
    """Wrapper around a given RL environment for a Network of ENUs model,
    turns reward into fitness and dumps relevant data"""


    def __init__(self, env, exp_name='maze'):
        self.env = env
        self.exp_name = exp_name
        self.n_output = self.env.n_actions
        #NOTE: +1 reward neuron
        self.n_input_neurons = self.env.n_obs + 1
        self.n_agents = self.env.n_agents

    def _convert_obs(self, obs, rewards):
        n_input_channels_used = 3
        X = np.zeros((self.n_agents, self.n_input_neurons, n_input_channels_used))
        #X[:, :obs.shape[1], 0] = obs
        # Shuffle only obs to avoid topology exploitation, reward neuron linked to EnuGlobal synapse connectivity
        X[:, :obs.shape[1], 0] = np.take_along_axis(obs, self.obs_shuffle, axis=1)
        # split pos and negative reward to different channels, And set to last input neuron
        if rewards is not None:
            X[rewards>0, -1, 1] = np.abs(rewards[rewards>0])
            X[rewards<=0, -1, 2] = np.abs(rewards[rewards<=0])
        return X

    def _convert_reward(self, obs, actions, rewards, infos, dones):
        fitness = np.copy(rewards)
        # first poison is considered positive reward, since learning to learn
        #NOTE: dead by env means less reward can be obtained so should implictely reduce overall fitness automatically
        fitness[np.logical_and(self._prev_reward_count == 1, rewards != 0)] = 1
        # include episode length as extra fitness, since not taking poison would allow survive longer, so should try avoid take poison
        fitness[dones==0] += 0.1/4
        return fitness

    def step(self, y):

        # if self.t % 3 != 0:
        #     actions = np.zeros((self.n_agents), dtype=np.int32) - 1
        # else:
            # winner take all, in given time window
        actions = y
            # if all same output, do nothing
            # equal_actions = self.y_hist.shape[1] == np.sum(self.y_hist == np.take_along_axis(self.y_hist, actions.reshape(-1, 1), axis=1), axis=-1)
            # actions[equal_actions] = -1
            # self.y_hist[:] = 0
        # take env step

        allobs, obs, rewards, dones, infos = self.env.step(actions)
        # X = self._convert_obs(obs, rewards)
        X=allobs
        self._prev_reward_count += rewards!=0
        fitness = self._convert_reward(obs, actions, rewards, infos, dones)
        self._prev_action = actions
        self._prev_obs = obs
        return X, rewards, fitness, None

    def reset(self):
        self.t = 0
        self.y_hist = np.zeros((self.n_agents, self.n_output), dtype=np.float32)
        self._prev_action = None
        self._prev_obs = None
        self._prev_reward_count = np.zeros((self.n_agents), dtype=np.float32)
        # each time different input/output neurons should have different meaning, to have learning to learn
        self.obs_shuffle = np.argsort(np.random.randn(self.n_agents, self.n_input_neurons - 1), axis=1, kind='mergesort')
        self.action_shuffle = np.argsort(np.random.randn(self.n_agents, self.n_output), axis=1, kind='mergesort')
        # reset env
        self.allobs,self.obs = self.env.reset()
        # return self._convert_obs(self.obs, None)
        return self.allobs

    def render(self):
        if self.t%4==0:
            self.env.render()

    def track_vis_data(self, vis_data, model, X, y_est, t):
        n_fetch = 128
        # TODO: also get our gates from the model
        vis_data+=[(X[:n_fetch, :], y_est[:n_fetch, :])]

    def dump_vis_data(self, vis_data, fitness_per_offspring, e):
        with open(get_data_path(e, self.exp_name, "output"), 'wb') as f:
            pickle.dump((vis_data, fitness_per_offspring), f)

    @staticmethod
    def load_vis_data(e, exp_name):
        with open(get_data_path(e, exp_name, "output"), 'rb') as f:
            vis_data, fitness_per_offspring = pickle.load(f)
        return vis_data, fitness_per_offspring

    @staticmethod
    def plot_vis_data(e, exp_name):
        vis_data, fitness_per_offspring = ExperimentEnvGlobalNetworkSurvival.load_vis_data(e, exp_name)