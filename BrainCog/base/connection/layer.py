import warnings
import math
import torch
from torch import nn
from torch import einsum
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from einops import rearrange


class VotingLayer(nn.Module):
    """
    用于SNNs的输出层, 几个神经元投票选出最终的类
    :param voter_num: 投票的神经元的数量, 例如 ``voter_num = 10``, 则表明会对这10个神经元取平均
    """

    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class WTALayer(nn.Module):
    """
    winner take all用于SNNs的每层后，将随机选取一个或者多个输出
    :param k: X选取的输出数目 k默认等于1
    """
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C,W,H]
        # ret.shape = [N, C,W,H]
        pos = x * torch.rand(x.shape, device=x.device)
        if self.k > 1:
            x = x * (pos >= pos.topk(self.k, dim=1)[0][:, -1:]).float()
        else:
            x = x * (pos >= pos.max(1, True)[0]).float()

        return x


class NDropout(nn.Module):
    """
    与Drop功能相同, 但是会保证同一个样本不同时刻的mask相同.
    """

    def __init__(self, p):
        super(NDropout, self).__init__()
        self.p = p
        self.mask = None

    def n_reset(self):
        """
        重置, 能够生成新的mask
        :return:
        """
        self.mask = None

    def create_mask(self, x):
        """
        生成新的mask
        :param x: 输入Tensor, 生成与之形状相同的mask
        :return:
        """
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return self.mask * x
        else:
            return x


class ThresholdDependentBatchNorm2d(_BatchNorm):
    """
    tdBN
    https://ojs.aaai.org/index.php/AAAI/article/view/17320
    """

    def __init__(self, alpha: float, threshold: float, *args, **kwargs):
        self.alpha = alpha
        self.threshold = threshold

        self.step = kwargs['step'] if 'step' in kwargs else 10
        self.layer_by_layer = kwargs['layer_by_layer'] if 'layer_by_layer' in kwargs else False
        kwargs.pop('step')
        kwargs.pop('layer_by_layer')

        kwargs['num_features'] = kwargs['num_features'] * self.step

        super().__init__(*args, **kwargs)

        assert self.layer_by_layer, \
            'tdBN may works in step-by-step mode, which will not take temporal dimension into batch norm'
        assert self.affine, 'ThresholdDependentBatchNorm needs to set `affine = True`!'

        torch.nn.init.constant_(self.weight, alpha * threshold)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        input = rearrange(input, '(t b) c w h -> b (t c) w h', t=self.step)
        output = super().forward(input)
        return rearrange(output, 'b (t c) w h -> (t b) c w h', t=self.step)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SMaxPool(nn.Module):
    """用于转换方法的最大池化层的常规替换
    选用具有最大脉冲发放率的神经元的脉冲通过，能够满足一般性最大池化层的需要

    Reference:
    https://arxiv.org/abs/1612.04052
    """

    def __init__(self, child):
        super(SMaxPool, self).__init__()
        self.opration = child
        self.sumspike = 0

    def forward(self, x):
        self.sumspike += x
        single = self.opration(self.sumspike * 1000)
        sum_plus_spike = self.opration(x + self.sumspike * 1000)

        return sum_plus_spike - single

    def reset(self):
        self.sumspike = 0


class LIPool(nn.Module):
    r"""用于转换方法的最大池化层的精准替换
    LIPooling通过引入侧向抑制机制保证在转换后的SNN中输出的最大值与期望值相同。

    Reference:
    https://arxiv.org/abs/2204.13271
    """

    def __init__(self, child=None):
        super(LIPool, self).__init__()
        if child is None:
            raise NotImplementedError("child should be Pooling operation with torch.")

        self.opration = child
        self.sumspike = 0

    def forward(self, x):
        self.sumspike += x
        out = self.opration(self.sumspike)
        self.sumspike -= F.interpolate(out, scale_factor=2, mode='nearest')
        return out

    def reset(self):
        self.sumspike = 0
