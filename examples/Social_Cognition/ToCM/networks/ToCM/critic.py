import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('/home/zhaofeifei/mambaSNN/networks/ToCM/')

from networks.ToCM.utils import build_model_snn, build_model
from networks.transformer.layers import AttentionEncoder
from braincog.base.node.node import LIFNode
from braincog.base.strategy.surrogate import AtanGrad

decay = 0.3
thresh = 0.3
lens = 0.25

# print("File Critic Here")
# 0.定义一个返回膜电势的 LIFNode
class BCNoSpikingLIFNode(LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dv: torch.Tensor):
        # print("dv: ", dv)
        # print("dv.shape: ", dv.shape)
        self.integral(dv)
        return self.mem


act_fun = AtanGrad(alpha=2., requires_grad=False)


def mem_update(fc, x, mem, spike):
    mem = mem * decay * (1 - spike) + fc(x)
    # spike = act_fun(mem)
    spike = act_fun(x=mem-1)
    return mem, spike


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_size, layers=2, node='LIFNode', time_window=16,
                 norm_in=True, output_style='voltage'):
        # hint是critic没有输出维度，action的输出维度是action的数量
        super().__init__()

        # 1.加入SNN的脉冲参数
        self._threshold = 0.5
        self.v_reset = 0.0
        self._time_window = time_window
        # 2.设置输出格式
        self.output_style = output_style
        # 3.ffn是否归一化
        self.norm = norm_in
        # if self.norm:
        #     self.in_norm = nn.BatchNorm1d(in_dim)
        #     self.in_norm.weight.data.fill_(1)
        #     self.in_norm.bias.data.zero_()
        # else:
        #     self.in_norm = lambda x: x
        self.in_norm = lambda x: x
        # 4.改变linear层的激活函数为LIFNode
        self.activation = node

        self.hidden_size = hidden_size
        self.layers = layers

        self.feedforward_model = build_model_snn(in_dim, 1, layers, hidden_size,
                                                 th=self._threshold, re=self.v_reset,
                                                 activation=self.activation, normalize=lambda x: x)
        # 这里feedforward的输出维度为1，其余一样 in_dim, out_dim, layers, hidden

        # 5. 定义输出神经元node
        if self.output_style == "sum":
            self.out_node = lambda x: x

        elif self.output_style == "voltage":
            self.out_node = BCNoSpikingLIFNode(tau=2.0)

    def forward(self, state_features, actions):
        # 6.加入脉冲步长模拟
        qs = []
        self.reset()  # why
        # 7.加入第一次输入的归一化，对最前面的输入进行norm
        state_features = self.in_norm(state_features)
        for t in range(self._time_window):
            x = self.feedforward_model(state_features)
            # 8.linear层之后还得有个node接住。否则如果对于ann来说，linear之后的浮点数就能作为最后的分值了，对于snn不行
            x = self.out_node(x)
            qs.append(x)

        if self.output_style == 'sum':
            value = sum(qs) / self._time_window
            return value
        elif self.output_style == 'voltage':
            value = qs[-1]
            return value

    # 调用modules里面node的n_reset
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

#SNN
# class MADDPGCritic(nn.Module):
#     def __init__(self, in_dim, hidden_size, node='nn.Tanh', time_window=1,  # time_window=16,
#                  norm_in=True, output_style='ann'):  # in_dim 1280 hidden_size 256
#         super().__init__()
#
#         # 1.加入SNN的脉冲参数
#         self._threshold = 0.5
#         self.v_reset = 0.0
#         self._time_window = time_window
#         # 2.设置输出格式
#         self.output_style = output_style
#         # 3. ffn是否归一化
#         self.norm = norm_in
#
#         # TODO no normalize
#         self.in_norm = lambda x: x
#         # 4.改变linear层的激活函数为LIFNode
#         self.activation = node  # TODO!!!!!!!!!!!
#
#         self.feedforward_model = build_model_snn(hidden_size, 1, 1, hidden_size,
#                                                  th=self._threshold, re=self.v_reset,
#                                                  activation=self.activation, normalize=lambda x: x)
#         # in_dim, out_dim, layers, hidden
#         # (in_dim = hidden)->hidden->hidden......-> (out_dim = 1)
#
#         self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
#         self.embed = nn.Linear(in_dim, hidden_size)  # 1280 256
#         self.prior = build_model_snn(in_dim, 1, 3, hidden_size,  # 1280 256
#                                      th=self._threshold, re=self._threshold,
#                                      activation=self.activation, normalize=lambda x: x)
#         # also in_dim, out_dim, layers, hidden
#         # (in_dim = hidden)->hidden->hidden......-> (out_dim = 1)
#         # 可能是个决策函数，决策优先选择哪个action
#
#         # 5. 定义输出神经元node
#         if self.output_style == "sum":
#             self.out_node = lambda x: x
#         elif self.output_style == "voltage":
#             self.out_node = BCNoSpikingLIFNode(tau=2.0)
#         elif self.output_style == 'ann':
#             self.out_node = lambda x: x
#
#     def forward(self, state_features, actions):
#         self.reset()  # reset函数得看看怎么加
#         n_agents = state_features.shape[-2]
#         batch_size = state_features.shape[:-2]
#         # 6.加入第一次输入的归一化
#         state_features = self.in_norm(state_features)
#         # 7.暂时不把编码加入模拟时长
#         embeds = F.relu(self.embed(state_features))
#         embeds = embeds.view(-1, n_agents, embeds.shape[-1])
#         attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
#
#         # 7.设置脉冲发放时长模拟，只在ffn层
#         qs = []
#         for t in range(self._time_window):
#             x = self.feedforward_model(attn_embeds)
#             x = self.out_node(x)
#             qs.append(x)
#
#         value = qs[-1]  # after 16 mem
#         # x = self.feedforward_model(attn_embeds)
#         # value = self.out_node(x)  # only mem once
#         return value
#
#     # 调用modules里面node的n_reset
#     def reset(self):
#         for mod in self.modules():
#             if hasattr(mod, 'n_reset'):
#                 mod.n_reset()
#ANN
class MADDPGCritic(nn.Module):
    def __init__(self, in_dim, hidden_size, layers=2, activation=nn.ELU):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.activation = activation
        self.feedforward_model = build_model(in_dim, 1, layers, hidden_size, activation)

    def forward(self, state_features, actions):
        return self.feedforward_model(state_features)
# critic_net = Critic(in_dim=2, hidden_size=32, layers=2, node='LIFNode', time_window=16, norm_in=True,
#                     output_style='voltage')
# print(critic_net)

# maddpg_critic_net = MADDPGCritic(in_dim=2, hidden_size=32, node='LIFNode', time_window=16, norm_in=True,
#                                  output_style='voltage')
# print(maddpg_critic_net)
