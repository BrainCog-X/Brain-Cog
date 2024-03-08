from agent.learners.ToCMLearner import ToCMLearner
from configs.ToCM.ToCMAgentConfig import ToCMConfig


# train->agent_configs = [ToCMLearnerConfig(ToCMConfig),] -> class ToCMConfig(Config) -> Config
class ToCMLearnerConfig(ToCMConfig):  # 从ToCMConfig继承，有输入维度、输出维度、隐层维度、隐层层数、动作维度、动作隐层维度、动作隐层层数、
    def __init__(self, env_name, RANDOM_SEED, device):
        super().__init__()
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 7e-4  # TODO
        self.VALUE_LR = 7e-4  # TODO
        self.CAPACITY = 500000
        self.MIN_BUFFER_SIZE = 100 
        self.MODEL_EPOCHS = 20  # TODO
        self.EPOCHS = 4  # TODO
        self.PPO_EPOCHS = 10  # TODO
        self.MODEL_BATCH_SIZE = 30#40
        self.BATCH_SIZE = 40
        self.SEQ_LENGTH = 50
        self.N_SAMPLES = 1
        self.TARGET_UPDATE = 128
        self.GRAD_CLIP = 100.0
        self.HORIZON = 15
        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 0.99998
        self.GRAD_CLIP_POLICY = 100.
        self.DEVICE = device  # TODO
        self.env_name = env_name    # TODO
        self.random_seed = RANDOM_SEED  # TODO
        self.num_agents = 2

    def create_learner(self):  # 通过config创建learner
        return ToCMLearner(self)
