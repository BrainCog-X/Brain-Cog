from configs.Config import Config
from env.starcraft.StarCraft import StarCraft
from env.mpe.MPE import MPE


class EnvConfig(Config):
    def __init__(self):
        pass

    def create_env(self):
        pass


# TODO
class MPEConfig(EnvConfig):
    def __init__(self, args):
        self.args = args

    def create_env(self):
        return MPE(self.args)  # an env object with base class MultiAgentEnv(gym.Env)


class StarCraftConfig(EnvConfig):
    def __init__(self, env_name, random_seed):
        self.env_name = env_name
        self.random_seed = random_seed  # TODO

    def create_env(self):
        return StarCraft(self.env_name, self.random_seed)


class EnvCurriculumConfig(EnvConfig):
    def __init__(self, env_configs, env_episodes, env_type, device, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_episodes = env_episodes  # （100，）
        self.ENV_TYPE = env_type  #
        self.device = device  # TODO

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculum(self.env_configs, self.env_episodes)


class EnvCurriculumSampleConfig(EnvConfig):
    def __init__(self, env_configs, env_probs, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_probs = env_probs

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculumSample(self.env_configs, self.env_probs)


class EnvCurriculumPrioritizedSampleConfig(EnvConfig):
    def __init__(self, env_configs, repeat_random_seed, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.repeat_random_seed = repeat_random_seed

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculumPrioritizedSample(self.env_configs, self.repeat_random_seed)
