import os
import sys

import numpy as np
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F


class CustomLinear(nn.Module):
    """
    用户自定义连接 通常stdp的计算
    """

    def __init__(self, weight, mask=None):
        super().__init__()

        self.weight = nn.Parameter(weight, requires_grad=True)
        self.mask = mask

    def forward(self, x: torch.Tensor):
        """
        :param x:输入 x.shape = [N ]
        """
        #
        # ret.shape = [C]

        return x.matmul(self.weight)

    def update(self, dw):
        """
        :param dw:权重更新量
        """
        with torch.no_grad():
            if self.mask is not None:
                dw *= self.mask
            self.weight.data += dw
