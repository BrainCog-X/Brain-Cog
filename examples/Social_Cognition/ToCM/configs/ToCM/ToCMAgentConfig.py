from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config

RSSM_STATE_MODE = 'discrete'


#
class ToCMConfig(Config):  # 从Config继承
    def __init__(self):
        super().__init__()
        self.HIDDEN = 64  # 隐藏层神经元个数
        self.MODEL_HIDDEN = 64  # 模型隐藏层神经元个数
        self.EMBED = 64  # 编码器神经元个数
        self.N_CATEGORICALS = 32  # 分类数
        self.N_CLASSES = 32  # 类别数
        self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES  # stochastic:随机的
        self.DETERMINISTIC = 64  # deterministic:确定的
        self.FEAT = self.STOCHASTIC + self.DETERMINISTIC  # feat:特征
        self.GLOBAL_FEAT = self.FEAT + self.EMBED  # global_feat:全局特征
        self.VALUE_LAYERS = 2  # value_layers:值层
        self.VALUE_HIDDEN = 64  # value_hidden:值隐藏层
        self.PCONT_LAYERS = 2  # pcont_layers:概率层
        self.PCONT_HIDDEN = 64  # pcont_hidden:概率隐藏层
        self.ACTION_SIZE = 9  # action_size:动作大小
        self.ACTION_LAYERS = 2  # action_layers:动作层
        self.ACTION_HIDDEN = 64  # action_hidden:动作隐藏层
        self.REWARD_LAYERS = 2  # reward_layers:奖励层
        self.REWARD_HIDDEN = 64  # reward_hidden:奖励隐藏层
        self.GAMMA = 0.99  # gamma:折扣因子
        self.DISCOUNT = 0.99  # discount:折扣
        self.DISCOUNT_LAMBDA = 0.95  # discount_lambda:折扣lambda
        self.IN_DIM = 30  # in_dim:输入维度
        self.LOG_FOLDER = 'wandb/'  # log_folder:日志文件夹
        self.num_agents = 2


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass

    def type(self):
        return None


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[RSSM_STATE_MODE]


