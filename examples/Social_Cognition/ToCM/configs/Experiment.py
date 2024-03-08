from configs.Config import Config


class Experiment(Config):  # 这个还没改且里面没有
    def __init__(self, steps, episodes, random_seed, env_config, controller_config, learner_config):
        super(Experiment, self).__init__()  # TODO  device 在env里面加入了
        self.steps = steps
        self.episodes = episodes
        self.random_seed = random_seed
        self.env_config = env_config
        self.controller_config = controller_config
        self.learner_config = learner_config
