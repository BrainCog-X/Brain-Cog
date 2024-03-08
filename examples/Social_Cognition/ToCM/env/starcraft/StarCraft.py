from smac.env import StarCraft2Env  # import a package smac


class StarCraft:

    def __init__(self, env_name, random_seed):
        # map_name ->
        self.env = StarCraft2Env(map_name=env_name, seed=random_seed, continuing_episode=True, difficulty="7")  # TODO
        env_info = self.env.get_env_info()

        self.n_obs = env_info["obs_shape"]
        self.n_actions = env_info["n_actions"]
        self.n_agents = env_info["n_agents"]

    def to_dict(self, l):
        return {i: e for i, e in enumerate(l)}

    def step(self, action_dict):
        reward, done, info = self.env.step(action_dict)
        return self.to_dict(self.env.get_obs()), {i: reward for i in range(self.n_agents)}, \
               {i: done for i in range(self.n_agents)}, info

    def reset(self):
        self.env.reset()
        return {i: obs for i, obs in enumerate(self.env.get_obs())}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_avail_agent_actions(self, handle):
        return self.env.get_avail_agent_actions(handle)
