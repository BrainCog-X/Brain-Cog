from agent.controllers.ToCMController import ToCMController
from configs.ToCM.ToCMAgentConfig import ToCMConfig


# train->agent_configs = [ToCMControllerConfig(ToCMConfig),] -> class ToCMConfig(Config) -> Config
class ToCMControllerConfig(ToCMConfig):
    def __init__(self, env_name, RANDOM_SEED, device):  # RANDOM_SEED:23 device:'cuda:6' env_name:'3s5z_vs_3s6z'
        super().__init__()

        self.EXPL_DECAY = 0.9999  # exploration decay rate：探索衰减率
        self.EXPL_NOISE = 0.  # exploration noise：探索噪声
        self.EXPL_MIN = 0.  # minimum exploration：最小探索
        self.DEVICE = device  # TODO
        self.env_name = env_name    # TODO
        self.random_seed = RANDOM_SEED  # TODO

    def create_controller(self):
        return ToCMController(self)
