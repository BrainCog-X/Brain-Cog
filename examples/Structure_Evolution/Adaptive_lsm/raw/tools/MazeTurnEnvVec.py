import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tools.Tools import save_fig, get_data_path

# np.random.seed(0)

class MazeTurnEnvVec:
    """Vectorized RL T-Maze environment written in pure Numpy. We require an efficient environment since we need to evaluate
    and run up to thousands of offspring in parallel"""

    def __init__(self, n_agents, n_steps):
        # 4 important points, start point, decision point, food point, dead point.
        # just generate a very large matrix that could fit any maze of any size, then can generate smaller maze as well
        self.n_actions = 3
        self.n_obs = 3
        self.max_size = 7
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.window = plt.figure()
        self.t_maze = True
        self.turn_based = False
        # steps can be longer if poison and need to turn around
        self.steps_to_food = 2
        if self.t_maze:
            self.steps_to_food = 3
        self.steps_to_food += self.steps_to_food*2
        # give some extra leniency
        self.steps_to_food *= 2
    def step(self, actions):
        # L R U D
        # TODO: check legal action or not..
        pos_copy = np.copy(self.agents_pos)
        actions = np.copy(actions)

        # GIVE TIME UPDATE WEIGHTS
        # actions[self.agents_reset > 0] = -1
        # actions[self.agent_energy<=0] = -1
        self.agents_reset[self.agents_reset > 0] -= 1
        # if turn based
        if self.turn_based:
            Forward = actions == 0
            self.agents_pos[np.logical_and(Forward, self.agent_directions == 0), 1] += 1
            self.agents_pos[np.logical_and(Forward, self.agent_directions == 2), 1] -= 1
            self.agents_pos[np.logical_and(Forward, self.agent_directions == 1), 0] -= 1
            self.agents_pos[np.logical_and(Forward, self.agent_directions == 3), 0] += 1
            L = actions == 1
            self.agent_directions[L] += 1
            R = actions == 2
            self.agent_directions[R] -= 1
            self.agent_directions[self.agent_directions > 3] = 0
            self.agent_directions[self.agent_directions < 0] = 3
        else:
            # or just direct movement
            U = actions == 2
            D = actions == 1
            R = actions == 0
            if self.agents_pos[U].size>0:
                self.agents_pos[U, 0] += 1
                self.agent_directions[U] = 3
            if self.agents_pos[D].size>0:
                self.agents_pos[D, 0] -= 1
                self.agent_directions[D] = 1
            if self.t_maze and self.agents_pos[R].size>0:
                self.agents_pos[R, 1] += 1
                self.agent_directions[R] = 0

        # UNDO MOVES THAT GOT AGENT INTO WALL
        self.current_cells = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1]]
        self.agents_pos[self.current_cells==1] = pos_copy[self.current_cells==1]
        movement_loss = np.prod(self.agents_pos==pos_copy, axis=-1)
        # CHECK IF FOOD CONSUMED, is reward + pos reset
        consumed_food = np.prod(self.agents_pos==self.food_pos[:, 0, :2], axis=-1).astype(np.bool)
        consumed_pois = np.prod(self.agents_pos==self.food_pos[:, 1, :2], axis=-1).astype(np.bool)
        self.consumed_count += consumed_food.astype(np.int32)
        self.consumed_count_total += consumed_food.astype(np.int32)
        self.consumed_count_pois += consumed_pois.astype(np.int32)
        self._reset_pos(np.logical_or(consumed_food, consumed_pois))
        # self._reset_pos_pois(consumed_pois)
        # self._reset_food(self.consumed_count==self.swap_limit, prob=0.0)
        # reset food for agents that ate food, and swap with some probability
        self._reset_food(self.consumed_count==5, prob=0.5)
        self.rewards = consumed_food.astype(np.float32) - consumed_pois.astype(np.float32) #* 0.5 #- movement_loss.astype(np.float32) * 0.01
        # get observation from current position of each agent
        self.agent_allobs,self.obs = self._get_obs_from_pos()
        # instant dead on second poison
        self.agent_energy += np.where(self.consumed_count_pois<=1, np.abs(self.rewards) * self.steps_to_food, -self.agent_energy * np.abs(self.rewards))
        # energy decay to encourage exploration, agent dies if running out of energy
        self.agent_energy = np.minimum(self.agent_energy, self.steps_to_food)
        self.agent_energy -= 1.0/4
        dones = self.agent_energy<=0

        return self.agent_allobs,self.obs, self.rewards, dones, None

    def _reset_pos(self, idxs):
        self.agents_pos[idxs] = [self.start_point, 2]  # set X pos
        self.agent_directions[idxs] = 0
        if self.max_size==5:
            self.agent_directions[idxs] = 1
        self.agents_reset[idxs] = 0
        self.agents_reset_count[idxs] += 1

    def _reset_pos_pois(self, idxs):
        #NOTE: 2 since we call reset twice!
        #NOTE: turning around already cost 8 steps, then 4x4 more is 16+8, 24, so reset should be much worse
        self.agents_reset[np.logical_and(idxs, self.agents_reset_count>2)] = 64

    def _reset_food(self, idxs, prob=0.5):
        # swap food with some probability, avoids agent overfitting on environment
        swap = np.take_along_axis(self.random_swap_matrix, self.consumed_count_total.reshape(-1, 1), axis=1).ravel()
        swap_idxs = swap * idxs
        food_loc = np.copy(self.food_pos[swap_idxs, 0, :])
        pois_loc = np.copy(self.food_pos[swap_idxs, 1, :])
        self.food_pos[swap_idxs, 0, :] = pois_loc
        self.food_pos[swap_idxs, 1, :] = food_loc
        # set maze value
        self.mazes[np.arange(self.mazes.shape[0]), self.food_pos[:, 0, 0], self.food_pos[:, 0, 1]] = 2
        self.mazes[np.arange(self.mazes.shape[0]), self.food_pos[:, 1, 0], self.food_pos[:, 1, 1]] = 3
        self.consumed_count[idxs] = 0

    def reset(self):
        self.consumed_count = np.zeros((self.n_agents), dtype=np.int32)
        self.consumed_count_total = np.zeros_like(self.consumed_count)
        # consistent swapping such that if agent eat food once for all agents swapped with same seed, fair fitness comparison
        max_eat = self.n_steps
        self.random_swap_matrix = np.random.uniform(0, 1, size=(1, max_eat)) >= 0.5
        self.random_swap_matrix = np.repeat(self.random_swap_matrix, int(self.n_agents), axis=0)

        self.agent_energy = np.zeros((self.n_agents), dtype=np.float32) + self.steps_to_food
        self.consumed_count_pois = np.zeros_like(self.consumed_count)
        #self.swap_limit = np.random.randint(1, 5, size=1)
        #self.swap_limit = np.random.randint(1, 4, size=self.n_agents)
        self.mazes = np.ones((self.n_agents, self.max_size, self.max_size), dtype=np.int32)
        #TODO: support variable maze length
        self.start_point = int(self.max_size/2)
        if self.t_maze:
            self.mazes[:, self.start_point, 2:-1] = 0
            self.mazes[:, 1:-1, -2] = 0
            # FOOD either at -1,-1 or -1,1?
            # two foods: x, y, value
            self.food_pos = np.zeros((self.n_agents, 2, 2), dtype=np.int32)
            self.food_pos[:, :, 1] = self.max_size - 2
            self.food_pos[:, 1, 0] = 1
            self.food_pos[:, 0, 0] = self.max_size - 2
            self._reset_food(np.ones(self.food_pos.shape[0], dtype=np.bool), prob=0.5)
        else:
            self.mazes[:, 1:-1, 1] = 0
            # two foods: x, y, value
            self.food_pos = np.zeros((self.n_agents, 2, 2), dtype=np.int32)
            self.food_pos[:, :, 1] = 1
            self.food_pos[:, 0, 0] = 1
            self.food_pos[:, 1, 0] = (self.max_size - 2)
            self._reset_food(np.ones(self.food_pos.shape[0], dtype=np.bool), prob=0.5)
        # AGENT
        self.agents_pos = np.ones((self.n_agents, 2), dtype=np.int32)
        self.agents_reset = np.zeros((self.n_agents), dtype=np.int32)
        self.agents_reset_count = np.zeros_like(self.agents_reset)
        self.agent_directions = np.zeros((self.n_agents), dtype=np.int32)
        self._reset_pos(np.arange(self.agents_pos.shape[0]))
        # OBS
        self.agent_allobs,self.obs = self._get_obs_from_pos()
        return self.agent_allobs,self.obs

    def _get_obs_from_pos(self):
        # obs is neighbouring cell states around agent
        obs = np.zeros((self.n_agents, self.n_obs), dtype=np.float32)
        raw_obs = np.zeros(self.n_agents, dtype=np.int32)
        # get observation based on direction agent is facing
        leftobs = np.zeros(self.n_agents)
        rightobs = np.zeros(self.n_agents)
        backobs = np.zeros(self.n_agents)

        # get observation based on direction agent is facing
        D = self.agent_directions == 0
        raw_obs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1] + 1][D]
        leftobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0] - 1, self.agents_pos[:, 1]][D]
        rightobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0] + 1, self.agents_pos[:, 1]][D]
        backobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1]-1][D]


        D = self.agent_directions == 2
        raw_obs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1] - 1][D]
        leftobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0] + 1, self.agents_pos[:, 1]][D]
        rightobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0] - 1, self.agents_pos[:, 1]][D]
        backobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1]+1][D]

        D = self.agent_directions == 1
        raw_obs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0] - 1, self.agents_pos[:, 1]][D]
        leftobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1] - 1][D]
        rightobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1] + 1][D]
        backobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0]+1, self.agents_pos[:, 1]][D]

        D = self.agent_directions == 3
        raw_obs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0] + 1, self.agents_pos[:, 1]][D]
        leftobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1] + 1][D]
        rightobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0], self.agents_pos[:, 1] - 1][D]
        backobs[D] = self.mazes[np.arange(self.mazes.shape[0]), self.agents_pos[:, 0]-1, self.agents_pos[:, 1]][D]

        # mark what was observed at different index
        obs[raw_obs == 1, 0] = 1
        obs[raw_obs == 2, 1] = 1
        obs[raw_obs == 3, 2] = 1
        allobs=np.squeeze(np.dstack((leftobs,raw_obs,rightobs,backobs)))
        return allobs, obs

    def render(self):
        plt.clf()
        sns.set_style("white")
        #TODO: support render all mazes? can reshape to square?
        max_render = 1
        flattened_render = np.dstack(np.split(self.mazes[18, :], max_render, axis=0)).reshape(self.mazes.shape[1], -1)
        flattened_render[flattened_render>1] = 0
        plt.axis('off')

        plt.imshow(flattened_render,cmap='bone')

        for j in range(1):
            i=18
            marker = ">"
            if self.agent_directions[i] == 1:
                marker = "^"
            if self.agent_directions[i] == 2:
                marker = "<"
            if self.agent_directions[i] == 3:
                marker = "v"
            obs_color = "black"
            if self.obs[i, 0] == 1:
                obs_color = "gray"
            if self.obs[i, 1] == 1:
                obs_color = "green"
            if self.obs[i, 2] == 1:
                obs_color = "red"
            alpha = 1
            if self.agent_energy[i]<=0:
                alpha = 1
            plt.scatter(self.agents_pos[i, 1] + j * self.mazes.shape[1], self.agents_pos[i, 0], color="skyblue", alpha=alpha, marker=marker)
            plt.scatter(self.agents_pos[i, 1] + j * self.mazes.shape[1], self.agents_pos[i, 0], color=obs_color,alpha=alpha, marker=marker, s=3)
            plt.scatter(self.food_pos[i, 0, 1] + j * self.mazes.shape[1], self.food_pos[i, 0, 0], color="green", alpha=1, marker="o")
            plt.scatter(self.food_pos[i, 1, 1] + j * self.mazes.shape[1], self.food_pos[i, 1, 0], color="red", alpha=1, marker="o")
        plt.pause(0.001)
        #plt.pause(2)

    @staticmethod
    def load_vis_data(e, exp_name):
        with open(get_data_path(e, exp_name, "output"), 'rb') as f:
            vis_data, fitness_per_offspring = pickle.load(f)
        return vis_data, fitness_per_offspring

    @staticmethod
    def plot_vis_data(e, exp_name):
        import matplotlib.pyplot as plt
        from cycler import cycler
        import seaborn as sns
        sns.set_style("whitegrid")

        vis_data, fitness_per_offspring = MazeTurnEnvVec.load_vis_data(e, exp_name)

        offspring_idx = 0
        #x, y_est, y = np.array(vis_data).transpose(1, 2, 0, 3)
        X, Y_est = map(np.array, zip(*vis_data))
        X_base, y_est_base = X[:, offspring_idx], Y_est[:, offspring_idx]
        #X_base, y_est_base = X_base[:300], y_est_base[:300]
        #X_base = np.max(X_base, axis=1)
        # --- normal output single example---
        plt.rc('axes', prop_cycle=(cycler('color', ['gray', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf'])))
        # OLD METHOD!
        plt.figure()
        #NOTE: last neuron is always reward neuron
        plt.plot(-X_base[:, :-1, 0], label="N-ENUs input", alpha=0.7)
        plt.plot(- np.max(X_base, axis=1)[:, 1], label="Positive reward", alpha=0.7, color='#2ca02c')
        plt.plot(- np.max(X_base, axis=1)[:, 2], label="Negative reward", alpha=0.7, color='#d62728')
        #plt.gca().set_color_cycle(['orange', 'purple', 'brown'])
        plt.plot(y_est_base[:, :], label="N-ENUs output", alpha=0.8)
        plt.legend(loc='upper right')
        save_fig(e, exp_name, "single_episode")

        plt.rc('axes', prop_cycle=(cycler('color', ['gray', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf'])))
        # new method!
        fig, grid = plt.subplots(2, sharex=True)
        # input
        #grid[1].set_prop_cycle(cycler('color', ['gray', '#ff7f0e', '#1f77b4']))
        grid[1].set_prop_cycle(cycler('color', ['gray', '#F5B041', '#2E86C1']))
        grid[1].plot(X_base[:, :-1, 0], label="N-ENUs input", alpha=0.7, linewidth=2)
        grid[1].plot(np.max(X_base, axis=1)[:, 1], label="Positive reward", alpha=0.7, color='#1ABC9C', linewidth=2)
        grid[1].plot(np.max(X_base, axis=1)[:, 2], label="Negative reward", alpha=0.7, color='#CB4335', linewidth=2)
        grid[1].legend(['Sensor (wall)', 'Sensor (red)', 'Sensor (green)', 'Positive reward', 'Negative reward'], loc='upper right')
        grid[1].set_ylabel('Neuron output')
        # output
        grid[0].set_prop_cycle(cycler('color', ['#9467bd', '#e377c2', '#17becf','#8c564b']))
        grid[0].plot(y_est_base[:, :], label="N-ENUs output", alpha=0.8, linewidth=2)
        grid[0].legend(['ENU-NN (left)', 'ENU-NN (right)', 'ENU-NN (forward)'],loc='upper right')
        plt.xlabel('t')
        grid[0].set_ylabel('ENU neuron output')
        plt.xlim(-5, X_base.shape[0]+10)
        save_fig(e, exp_name, "single_episode_dual")
        #plt.show()

    @staticmethod
    def plot_rollout_data(e, exp_name):
        #TODO: dump rollout as array not the actual plots
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("white")
        import os

        rollout_path = "./" + get_data_path(e, exp_name, "rollout").split(".")[1][:-1]+"/"
        rollout_files = sorted(os.listdir(rollout_path))
        rollouts = []
        for i in range(4, 200, 4):
            file = "rollout_{}_.png".format(i)
            print(file)
            rollout = plt.imread(rollout_path + file)
            rollout = rollout[80:450, 200:500]
            rollout = np.where(rollout[:, :, [0]]> 0.98, 1, rollout)
            rollouts.append(rollout)
            # plt.imshow(rollout)
            # plt.show()
        # for rollout in rollouts:
        vert_bar = np.zeros((rollout.shape[0], 5, 4))
        vert_bar[::] += 0.5
        # get red -> learned go other way
        # food swapped -> sees red -> learned turn around
        rollouts1 = [rollouts[1], rollouts[9], rollouts[10], rollouts[11],  rollouts[17], rollouts[18]]#, rollouts[19]
        #, rollouts[28]
        #rollouts[24],
        rollouts2 = [rollouts[25], rollouts[29], rollouts[30], rollouts[31], rollouts[32], rollouts[33]]
        plt.axis('off')
        plt.imshow(np.vstack([np.column_stack(rollouts1), np.column_stack(rollouts2)]), cmap='gray')
        save_fig(e, exp_name, "rollout_combined")
        #plt.show()




if __name__ == '__main__':
    """Test function"""
    n_offspring = 1024
    envs = MazeTurnEnvVec(n_offspring, n_steps=400)
    envs.n_pseudo_env = 8
    while True:
        envs.reset()
        opt_actions_up = [0,0,0,0,1,0]
        opt_actions_down = [0,0,0,0,2,0]
        opt_actions = [opt_actions_up, opt_actions_down]
        opt_current = 0
        total_reward = 0
        rewards = np.zeros((n_offspring))
        rewards_all = np.zeros_like(rewards)
        k = 0
        n_steps = 200
        for i in range(n_steps):
            actions = np.random.randint(0, envs.n_actions, size=n_offspring)
            #actions[:] = opt_actions_up[i%len(opt_actions_up)]
            #actions[0] = opt_actions[opt_current][k % len(opt_actions_up)]
            obs, rewards,_,_ = envs.step(actions=actions)
            total_reward += (rewards[0] * 100) + 1
            rewards_all += (rewards * 100) + 1
            #print(rewards[0])
            #print(obs[0], rewards[0])

            envs.render()
            plt.pause(0.5)

            k += 1
            if rewards[0] < 0:
                #print(rewards_last[0])
                if opt_current==0:
                    opt_current = 1
                else:
                    opt_current = 0
            if rewards[0] != 0:
                k = 0
        total_reward/=n_steps
        rewards_all/=n_steps
        print(rewards_all[0], np.mean(rewards_all), np.std(rewards_all), np.std(rewards_all)/np.mean(rewards_all))
        print(rewards_all[:10])
