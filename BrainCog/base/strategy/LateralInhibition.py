import warnings
import torch
from torch import nn
import torch.nn.functional as F


class LateralInhibition(nn.Module):
    """
    侧抑制 用于发放脉冲的神经元抑制其他同层神经元 在膜电位上作用
    """
    def __init__(self, node, inh, mode="constant"):
        super().__init__()
        self.inh = inh
        self.node = node
        self.mode = mode

    def forward(self, x: torch.Tensor, xori=None):
        # x.shape = [N, C,W,H]
        # ret.shape = [N, C,W,H]
        if self.mode == "constant":

            self.node.mem = self.node.mem - self.inh * (x.max(1, True)[0] - x)

        elif self.mode == "max":
            self.node.mem = self.node.mem - self.inh * xori.max(1, True)[0] .detach() * (x.max(1, True)[0] - x)
        elif self.mode == "threshold":
            self.node.mem = self.node.mem - self.inh * self.node.threshold * (x.max(1, True)[0] - x)
        else:
            pass
        return x
