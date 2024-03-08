from copy import deepcopy
import numpy as np
import ray
import torch
from collections import defaultdict

from environments import Env


@ray.remote(num_gpus=1) # TODO
class ToCMWorker:

    def __init__(self, idx, env_config, controller_config):
        self.runner_handle = idx
        self.env = env_config.create_env()
        self.controller = controller_config.create_controller()  # controller
        self.in_dim = controller_config.IN_DIM
        self.env_type = env_config.ENV_TYPE
        self.controller_config = controller_config
        self.device = env_config.device

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT:
            return self.done[handle] == 0

        else:  # TODO
            return self.env.agents[handle].movable

    def _select_actions(self, state):
        avail_actions = []
        observations = []
        fakes = []

        nn_mask = None

        for handle in range(self.env.n_agents):
            if self.env_type == Env.STARCRAFT:
                avail_actions.append(torch.tensor(self.env.get_avail_agent_actions(handle)))

            if self._check_handle(handle) and handle in state:
                fakes.append(torch.zeros(1, 1))
                observations.append(state[handle].unsqueeze(0))
            elif self.done[handle] == 1:  # handle is not in state
                fakes.append(torch.ones(1, 1))  # fake move
                observations.append((self.get_absorbing_state()).to(self.device))
            else:
                fakes.append(torch.zeros(1, 1))
                obs = (torch.tensor(self.env.obs_builder._get_internal(handle)).float().unsqueeze(0)).to(self.device)
                observations.append(obs)

        # print("observations:", observations)
        observations = torch.cat(observations).unsqueeze(0)  # TODO
        # print("observations:", observations)
        av_action = torch.stack(avail_actions).unsqueeze(0).to(self.device) if len(avail_actions) > 0 else None
        # print("av_actions:", av_action)
        nn_mask = nn_mask.unsqueeze(0).repeat(8, 1, 1).to(self.device) if nn_mask is not None else None
        # print("nn_mask:", nn_mask)
        actions = self.controller.step(observations, av_action, nn_mask).to(self.device)
        # print("actions:", actions)
        return actions, observations, torch.cat(fakes).unsqueeze(0), av_action   # TODO use controller to model and pred

    def _wrap(self, d):
        for key, value in d.items():
            d[key] = torch.tensor(value).to(self.controller_config.DEVICE).float()
        return d

    def get_absorbing_state(self):
        state = torch.zeros(1, self.in_dim).to(self.device)  # TODO
        return state

    def augment(self, data, inverse=False):
        aug = []
        default = list(data.values())[0].reshape(1, -1)
        for handle in range(self.env.n_agents):
            if handle in data.keys():
                aug.append(data[handle].reshape(1, -1))
            else:
                aug.append(torch.ones_like(default) if inverse else torch.zeros_like(default))
        return torch.cat(aug).unsqueeze(0).to(self.device)  # TODO

    def _check_termination(self, info, steps_done):
        if self.env_type == Env.STARCRAFT or self.env_type == Env.MPE:
            return "episode_limit" not in info
        else:
            return steps_done < self.env.max_time_steps  # can not chao shi

    def run(self, ToCM_params):
        f"""
        interact with environment
        :param ToCM_params: 
        :return: rollout: dict reward steps_done
        """
        self.controller.receive_params(ToCM_params)
        # Share the parameters learned by the learner with the controller.
        # freeze the parameters

        state = self._wrap(self.env.reset())  # to device
        steps_done = 0
        self.done = defaultdict(lambda: False)
        episode_rewards = []
        while True:
            steps_done += 1
            # print("state=", state)
            actions, obs, fakes, av_actions = self._select_actions(state)  # use controller to select action
            if self.env_type == Env.MPE:
                next_state, reward, done, info = self.env.step(actions)  # use env to update, with cpu
                rewards = []
                for key, value in reward.items():
                    rewards.append(value)
                episode_rewards.append(rewards)
            else:
                next_state, reward, done, info = self.env.step([action.argmax() for i, action in enumerate(actions)])
            next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), \
                self._wrap(deepcopy(done))  # to device
            self.done = done
            self.controller.update_buffer({"action": actions,
                                           "observation": obs,
                                           "reward": self.augment(reward),
                                           "done":  self.augment(done),
                                           "fake": fakes,
                                           "avail_action": av_actions})

            state = next_state
            if all([done[key] == 1 for key in range(self.env.n_agents)]):
                # print("Done")
                if self._check_termination(info, steps_done):
                    # print("Done!")
                    obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                    actions = torch.zeros(1, self.env.n_agents, actions.shape[-1])
                    index = torch.randint(0, actions.shape[-1], actions.shape[:-1], device=actions.device)
                    actions.scatter_(2, index.unsqueeze(-1), 1.)
                    items = {"observation": obs,
                             "action": actions,
                             "reward": torch.zeros(1, self.env.n_agents, 1),
                             "fake": torch.ones(1, self.env.n_agents, 1),
                             "done": torch.ones(1, self.env.n_agents, 1),
                             "avail_action": torch.ones_like(actions) if self.env_type == Env.STARCRAFT else None}
                    self.controller.update_buffer(items)
                    self.controller.update_buffer(items)  # why two
                break
        if self.env_type == Env.MPE:
            reward = np.mean(np.sum(episode_rewards, axis=0))  # TODO
        else:
            reward = 1. if 'battle_won' in info and info['battle_won'] else 0.
        return self.controller.dispatch_buffer(), {"idx": self.runner_handle,
                                                   "reward": reward,  # a num
                                                   "steps_done": steps_done}
