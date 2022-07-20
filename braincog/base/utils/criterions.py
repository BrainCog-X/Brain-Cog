import numpy as np
import torch


class UnilateralMse(torch.nn.Module):
    """
    扩展单边的MSE损失, 用于控制输出层的期望fire-rate 高于 thresh
    :param thresh: 输出层的期望输出频率
    """
    def __init__(self, thresh=1.):
        super(UnilateralMse, self).__init__()
        self.thresh = thresh
        self.loss = torch.nn.MSELoss()

    def forward(self, x, target):
        # x = nn.functional.softmax(x, dim=1)
        torch.clip(x, max=self.thresh)
        if x.shape == target.shape:
            return self.loss(x, target)
        return self.loss(x, torch.zeros_like(x).scatter_(1, target.view(-1, 1), self.thresh))


class MixLoss(torch.nn.Module):
    """
    混合损失函数, 可以将任意的损失函数与UnilateralMse损失混合
    :param ce_loss: 任意的损失函数
    """
    def __init__(self, ce_loss):
        super(MixLoss, self).__init__()
        self.ce = ce_loss
        self.mse = UnilateralMse(1.)

    def forward(self, x, target):
        return 0.1 * self.ce(x, target) + self.mse(x, target)
