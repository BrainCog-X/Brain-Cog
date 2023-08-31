import numpy as np
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from braincog.model_zoo.base_module import DeformConvPack
from braincog.model_zoo.base_module import BaseLinearModule


# from mmcv.ops import ModulatedDeformConv2dPack


def si_relu(x, positive):
    if positive == 1:
        return torch.where(x > 0., x, torch.zeros_like(x))
    elif positive == 0:
        return x
    elif positive == -1:
        return torch.where(x < 0., x, torch.zeros_like(x))
    else:
        raise ValueError


class SiReLU(nn.Module):
    def __init__(self, positive=0):
        super().__init__()
        self.positive = positive

    def forward(self, x):
        return si_relu(x, self.positive)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal(m.weight.data, gain=0.1)
        torch.nn.init.constant(m.bias.data, 0.)

OPS_Mlp = {
    'mlp': lambda C, act_fun:
        SiMLP(C, C, act_fun=act_fun, positive=0),
    'mlp_p': lambda C, act_fun:
        SiMLP(C, C, act_fun=act_fun, positive=1),
    'mlp_n': lambda C, act_fun:
        SiMLP(C, C, act_fun=act_fun, positive=-1),

    'skip_connect': lambda C, act_fun:
        Identity(positive=0),
    'skip_connect_p': lambda C, act_fun:
        Identity(positive=1),
    'skip_connect_n': lambda C, act_fun:
        Identity(positive=-1),
}

OPS = {
    'avg_pool_3x3': lambda C, stride, affine, act_fun: nn.AvgPool2d(3, stride=stride, padding=1,
                                                                    count_include_pad=False),
    'conv_3x3': lambda C, stride, affine, act_fun:
        ReLUConvBN(C_in=C, C_out=C, kernel_size=3, padding=1, stride=stride, affine=affine, act_fun=act_fun, positive=0),
    'conv_5x5': lambda C, stride, affine, act_fun:
        ReLUConvBN(C_in=C, C_out=C, kernel_size=5, padding=2, stride=stride, affine=affine, act_fun=act_fun, positive=0),
    'max_pool_3x3': lambda C, stride, affine, act_fun: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine, act_fun:
        Identity(positive=0) if stride == 1 else FactorizedReduce(C, C, affine=affine, act_fun=act_fun),
    'sep_conv_3x3': lambda C, stride, affine, act_fun:
        SepConv(C, C, 3, stride, 1, affine=affine, act_fun=act_fun, positive=0),
    'sep_conv_5x5': lambda C, stride, affine, act_fun:
        SepConv(C, C, 5, stride, 2, affine=affine, act_fun=act_fun, positive=0),
    'sep_conv_7x7': lambda C, stride, affine, act_fun:
        SepConv(C, C, 7, stride, 3, affine=affine, act_fun=act_fun, positive=0),
    'dil_conv_3x3': lambda C, stride, affine, act_fun:
        DilConv(C, C, 3, stride, 2, 2, affine=affine, act_fun=act_fun, positive=0),
    'dil_conv_5x5': lambda C, stride, affine, act_fun:
        DilConv(C, C, 5, stride, 4, 2, affine=affine, act_fun=act_fun, positive=0),
    'def_conv_3x3': lambda C, stride, affine, act_fun:
        DeformConv(C, C, 3, stride, 1, affine=affine, act_fun=act_fun, positive=0),
    'def_conv_5x5': lambda C, stride, affine, act_fun:
        DeformConv(C, C, 5, stride, 2, affine=affine, act_fun=act_fun, positive=0),

    'avg_pool_3x3_p': lambda C, stride, affine, act_fun: nn.Sequential(
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
        SiReLU(positive=1)
    ),
    'max_pool_3x3_p': lambda C, stride, affine, act_fun: nn.Sequential(
        nn.MaxPool2d(3, stride=stride, padding=1),
        SiReLU(positive=1)
    ),
    'conv_3x3_p': lambda C, stride, affine, act_fun:
        ReLUConvBN(C_in=C, C_out=C, kernel_size=3, padding=1, stride=stride, affine=affine, act_fun=act_fun, positive=1),
    'conv_5x5_p': lambda C, stride, affine, act_fun:
        ReLUConvBN(C_in=C, C_out=C, kernel_size=5, padding=2, stride=stride, affine=affine, act_fun=act_fun, positive=1),
    'skip_connect_p': lambda C, stride, affine, act_fun:
        Identity(positive=1) if stride == 1 else FactorizedReduce(C, C, affine=affine, act_fun=act_fun, positive=1),
    'sep_conv_3x3_p': lambda C, stride, affine, act_fun:
        SepConv(C, C, 3, stride, 1, affine=affine, act_fun=act_fun, positive=1),
    'sep_conv_5x5_p': lambda C, stride, affine, act_fun:
        SepConv(C, C, 5, stride, 2, affine=affine, act_fun=act_fun, positive=1),
    'sep_conv_7x7_p': lambda C, stride, affine, act_fun:
        SepConv(C, C, 7, stride, 3, affine=affine, act_fun=act_fun, positive=1),
    'dil_conv_3x3_p': lambda C, stride, affine, act_fun:
        DilConv(C, C, 3, stride, 2, 2, affine=affine, act_fun=act_fun, positive=1),
    'dil_conv_5x5_p': lambda C, stride, affine, act_fun:
        DilConv(C, C, 5, stride, 4, 2, affine=affine, act_fun=act_fun, positive=1),
    'def_conv_3x3_p': lambda C, stride, affine, act_fun:
        DeformConv(C, C, 3, stride, 1, affine=affine, act_fun=act_fun, positive=1),
    'def_conv_5x5_p': lambda C, stride, affine, act_fun:
        DeformConv(C, C, 5, stride, 2, affine=affine, act_fun=act_fun, positive=1),

    'avg_pool_3x3_n': lambda C, stride, affine, act_fun: nn.Sequential(
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
        SiReLU(positive=-1)
    ),
    'max_pool_3x3_n': lambda C, stride, affine, act_fun: nn.Sequential(
            nn.MaxPool2d(3, stride=stride, padding=1),
            SiReLU(positive=-1)
    ),
    'conv_3x3_n': lambda C, stride, affine, act_fun:
        ReLUConvBN(C_in=C, C_out=C, kernel_size=3, padding=1, stride=stride, affine=affine, act_fun=act_fun, positive=-1),
    'conv_5x5_n': lambda C, stride, affine, act_fun:
        ReLUConvBN(C_in=C, C_out=C, kernel_size=5, padding=2, stride=stride, affine=affine, act_fun=act_fun, positive=-1),
    'skip_connect_n': lambda C, stride, affine, act_fun:
        Identity(positive=-1) if stride == 1 else FactorizedReduce(C, C, affine=affine, act_fun=act_fun, positive=-1),
    'sep_conv_3x3_n': lambda C, stride, affine, act_fun:
        SepConv(C, C, 3, stride, 1, affine=affine, act_fun=act_fun, positive=-1),
    'sep_conv_5x5_n': lambda C, stride, affine, act_fun:
        SepConv(C, C, 5, stride, 2, affine=affine, act_fun=act_fun, positive=-1),
    'sep_conv_7x7_n': lambda C, stride, affine, act_fun:
        SepConv(C, C, 7, stride, 3, affine=affine, act_fun=act_fun, positive=-1),
    'dil_conv_3x3_n': lambda C, stride, affine, act_fun:
        DilConv(C, C, 3, stride, 2, 2, affine=affine, act_fun=act_fun, positive=-1),
    'dil_conv_5x5_n': lambda C, stride, affine, act_fun:
        DilConv(C, C, 5, stride, 4, 2, affine=affine, act_fun=act_fun, positive=-1),
    'def_conv_3x3_n': lambda C, stride, affine, act_fun:
        DeformConv(C, C, 3, stride, 1, affine=affine, act_fun=act_fun, positive=-1),
    'def_conv_5x5_n': lambda C, stride, affine, act_fun:
        DeformConv(C, C, 5, stride, 2, affine=affine, act_fun=act_fun, positive=-1),

    'conv_7x1_1x7': lambda C, stride, affine, act_fun: nn.Sequential(
        # nn.ReLU(inplace=False),
        act_fun(),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride),
                  padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1),
                  padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
    'transformer': lambda C, stride, affine, act_fun:
        FactorizedReduce(
            C, C, affine=affine, act_fun=act_fun) if stride != 1 else TransformerEncoderLayer(C),
}


class SiMLP(nn.Module):
    def __init__(self, c_in, c_out, act_fun=nn.ReLU, positive=0, *args, **kwargs):
        super(SiMLP, self).__init__()
        self.op = nn.Sequential(
            nn.Linear(c_in, c_out, bias=True),
            act_fun()
        )
        self.positive = positive

    def forward(self, x):
        out = self.op(si_relu(x, self.positive))
        return out


class ReLUConvBN(nn.Module):
    """
    ReLu -> Conv2d -> BatchNorm2d
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, act_fun=nn.ReLU, positive=0):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            # nn.ReLU(inplace=False),
            # act_fun(),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        self.positive = positive
        # if positive == -1:
        #     weight_init(self.op)

    def forward(self, x):
        out = self.op(x)
        return si_relu(out, self.positive)


class DilConv(nn.Module):
    """
    Dilation Convolution ： ReLU -> DilConv -> Conv2d -> BatchNorm2d
    """

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, act_fun=nn.ReLU, positive=0):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            # nn.ReLU(inplace=False),
            act_fun(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        self.positive = positive
        # if positive == -1:
        #     weight_init(self.op)

    def forward(self, x):
        out = self.op(x)
        return si_relu(out, self.positive)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, act_fun=nn.ReLU, positive=0):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # nn.ReLU(inplace=False),
            act_fun(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size,
                      stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
        self.positive = positive
        # if positive == -1:
        #     weight_init(self.op)

    def forward(self, x):
        out = self.op(x)
        return si_relu(out, self.positive)


class Identity(nn.Module):

    def __init__(self, positive=0):
        super(Identity, self).__init__()
        self.positive = positive

    def forward(self, x):
        return si_relu(x, self.positive)


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)  # N * C * W * H


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True, act_fun=nn.ReLU, positive=0):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        # self.relu = nn.ReLU(inplace=False)
        self.activation = act_fun()
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 3,
                                stride=2, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 3,
                                stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
        self.positive = positive
        # if positive == -1:
        #     weight_init(self.op)

    def forward(self, x):
        # x = self.relu(x)
        x = self.activation(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        out = si_relu(out, self.positive)
        return out


class DeformConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, act_fun=nn.ReLU, positive=0):
        super(DeformConv, self).__init__()
        self.op = nn.Sequential(
            # nn.ReLU(inplace=False),
            act_fun(),
            DeformConvPack(C_in, C_out, kernel_size=kernel_size,
                           stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(C_out, affine=affine)
        )
        self.positive = positive
        # if positive == -1:
        #     weight_init(self.op)

    def forward(self, x):
        out = self.op(x)
        return si_relu(out, self.positive)


class Attention(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=4, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(self, d_model, nhead=4, dim_feedforward=256, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)
        dim_feedforward = d_model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # print(src.shape)
        c = src.shape[-1]
        src = rearrange(src, 'b d r c -> b (r c) d')
        # print(src.shape)
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        src = rearrange(src, 'b (r c) d -> b d r c', c=c)
        return src


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
