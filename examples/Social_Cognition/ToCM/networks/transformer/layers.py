import numpy as np
import torch
import torch.nn as nn


#  位置编码
class PositionalEncoding(nn.Module):
    __author__ = "Yu-Hsiang Huang"

    def __init__(self, d_hid, n_position=2):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        '''
        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state.
        input: buffer's name, buffer's shape 应该是隐藏层之类的
        '''
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        # return x + self.pos_table[:, :x.size(1)].clone().detach() 使用pos_table

    @staticmethod  # 系统提示我这个方法静态
    def _get_sinusoid_encoding_table(n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):  # 获取每个位置的角度向量
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # shape [pos_i, d_hid, position]
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # 增加一个维度

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class AttentionEncoder(nn.Module):

    def __init__(self, n_layers, in_dim, hidden, dropout=0.):
        super().__init__()
        self.pos_embed = PositionalEncoding(hidden, 30)  # 返回位置编码方案,维度++
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=in_dim, nhead=8,
                                                                        dim_feedforward=hidden,
                                                                        dropout=dropout), n_layers)

    def forward(self, enc_input, **kwargs):
        enc_input = self.pos_embed(enc_input)
        x = self.encoder(enc_input.permute(1, 0, 2), **kwargs)
        return x.permute(1, 0, 2)  # 混洗 调换顺序


class AttentionActorEncoder(nn.Module):

    def __init__(self, n_layers, in_dim, hidden, dropout=0.):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=in_dim, nhead=8,
                                                                        dim_feedforward=hidden,
                                                                        dropout=dropout), n_layers)

    def forward(self, enc_input, **kwargs):
        x = self.encoder(enc_input,  **kwargs)
        return x  # 混洗 调换顺序



