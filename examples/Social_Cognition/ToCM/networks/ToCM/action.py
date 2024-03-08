import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# if '/home/zhaofeifei/.local/lib/python3.8/site-packages' in sys.path:
#     sys.path.remove('/home/zhaofeifei/.local/lib/python3.8/site-packages')

# sys.path.append('/home/zhaofeifei/mambaSNN_Mpe/networks/ToCM/')
# sys.path.append("/home/zhaofeifei/mambaSNN_Mpe/")

from torch.distributions import OneHotCategorical
from networks.transformer.layers import AttentionEncoder, AttentionActorEncoder
from networks.ToCM.utils import build_model_snn, build_model
from braincog.base.node.node import LIFNode, BaseNode, PLIFNode, DoubleSidePLIFNode
from braincog.base.strategy.surrogate import AtanGrad


class BCNoSpikingLIFNode(LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dv: torch.Tensor):
        # print("dv: ", dv)
        # print("dv.shape: ", dv.shape)
        self.integral(dv)
        return self.mem

#SNN
# class Actor(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_size, layers, node='LIFNode', time_window=8,
#                  norm_in=True, output_style='voltage'):  # 1.激活函数需要改成node # voltage
#         super().__init__()
#         # 1.加入SNN的脉冲参数
#         self._threshold = 0.5
#         self.v_reset = 0.0
#         self.tau = 0.5
#         self._time_window = time_window
#         # 2.设置输出格式
#         self.output_style = output_style
#         # 3.ffn是否归一化
#         self.norm = norm_in
#         self.activation = node
#         self.feedforward_model = build_model_snn(in_dim, out_dim, layers, hidden_size,  # kkkk TODO!!!
#                                                  th=self._threshold, re=self.v_reset, tau=self.tau,
#                                                  activation=self.activation, normalize=lambda x: x)  # TODO
#         if self.output_style == 'ann':
#             self.out_node = lambda x: x
#         elif self.output_style == 'voltage':
#             self.out_node = BCNoSpikingLIFNode(tau=1.0)
#
#     def forward(self, state_features):
#         # 5.加入脉冲仿真步长
#         # print("state.shape", state_features.shape)
#         self.reset()  # why
#         for t in range(self._time_window):
#             x = self.feedforward_model(state_features)
#             x = self.out_node(x)
#         # print("x", x.shape)
#         action_dist = OneHotCategorical(logits=x)
#         action = action_dist.sample()  # 长度为x，一行默认 tensor([0., 1., 0., 0.])
#         return action, x
#
#     # 调用modules里面node的n_reset
#     def reset(self):
#         for mod in self.modules():
#             if hasattr(mod, 'n_reset'):
#                 mod.n_reset()

#ANN
class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()

        self.feedforward_model = build_model(in_dim, out_dim, layers, hidden_size, activation)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x

class AttentionActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, node='LIFNode', time_window=16,
                 norm_in=True, output_style='voltage'):  # 2.激活层
        super().__init__()
        # 1.加入SNN的脉冲参数
        self._threshold = 0.5
        self.v_reset = 0.0
        self._time_window = time_window
        # 2.设置输出格式
        self.output_style = output_style
        # 3.ffn是否归一化
        self.norm = norm_in
        # 4.改变linear层的激活函数为LIFNode
        self.activation = node

        # hint: hidden_size = 其他网络的in_dim
        self.feedforward_model = build_model_snn(hidden_size, out_dim, 2, hidden_size,
                                                 th=self._threshold, re=self.v_reset,
                                                 activation=self.activation, normalize=lambda x: x)  # TODO
        # build_model_snn(in_dim, out_dim, layers, hidden, activation, normalize=lambda x: x)
        self._attention_stack = AttentionActorEncoder(1, hidden_size, hidden_size)
        # no pos_embedding
        # self._attention_stack = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=in_dim, nhead=1,
        #                                                                         dim_feedforward=hidden_size,
        #                                                                         dropout=.0), num_layers=1)  # TODO
        # n_layers, in_dim, hidden
        # 使用transformer的编码器，加入位置编码，加入隐藏单元d_hid,返回一个序列,其中第0维度应该是观测变量？
        self.embed = nn.Linear(in_dim, hidden_size)
        self.node1 = LIFNode(threshold=self._threshold, v_reset=self.v_reset)
        self.node2 = LIFNode(threshold=self._threshold, v_reset=self.v_reset)
        # 5. 定义一个处理linear层的node
        if self.activation == 'LIFNode':
            if self.output_style == 'voltage':
                self.out_node = BCNoSpikingLIFNode(tau=2.0)

    def forward(self, state_features):  # 状态值tensor
        # print("state_feat:", state_features[0])
        # attn_embeds = self._attention_stack(state_features)
        # n_agents = state_features.shape[-2]  # 推测state的维度为[batch_size(m,n), n_agents, in_dim]
        # batch_size = state_features.shape[:-2]  # 除去最后2维度的维度
        qs = []
        self.reset()  # why
        # print("attn_embeds", attn_embeds[0])
        # print("state.shape", state_features.shape)
        attn_embeds = self.embed(state_features)  # Linear
        for t in range(self._time_window):
            embeds = self.node1(attn_embeds)  # Node
            # attn_embeds = embeds.view(-1, n_agents, embeds.shape[-1])
            # embeds = self.node2(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
            x = self.feedforward_model(embeds)
            x = self.out_node(x)
            qs.append(x)

        p = torch.zeros(qs[0].shape)
        if self.output_style == "sum":
            p = sum(qs) / self._time_window
        elif self.output_style == "voltage":
            p = qs[-1]  # TODO

        # p = F.softmax(p)
        # print("pi:", p[0])
        action_dist = OneHotCategorical(logits=p)  # 编码器，长度为p

        action = action_dist.sample()
        # print("actions", action[0])
        # 对输出进行采样
        return action, p  # 返回一个行动序列action为每个位置符合p = x[i]的0,1序列

    # 调用modules里面node的n_reset
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

# aa = AttentionActor(16, 8, 64, 3)  # in_dim, out_dim, hidden_size, layers,
# state_feature = torch.randn([8, 8, 2, 16])  # 输入变量维度
# out, x = aa(state_feature)
# print(aa)
# print(out)
# print(out.shape)
