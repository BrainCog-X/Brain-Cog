import torch
import torch.nn as nn
import torch.nn.functional as F
from braincog.base.utils.criterions import UnilateralMse
from utils import num_ops, type_num, edge_num

__all__ = ['ConvSeparateLoss', 'TriSeparateLoss']


class MseSeparateLoss(nn.modules.loss._Loss):

    def __init__(self, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(MseSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight
        self.criterion = UnilateralMse(1.)

    def forward(self, input1, target1, input2):
        loss1 = self.criterion(input1, target1)
        loss2 = -F.mse_loss(input2, torch.tensor(0.5,
                            requires_grad=False).cuda())
        return loss1 + self.weight * loss2, loss1.item(), loss2.item()


class ConvSeparateLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L2"""

    def __init__(self, loss1_fn, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ConvSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight
        self.loss1_fn = loss1_fn

    def forward(self, input1, target1, input2):
        loss1 = self.loss1_fn(input1, target1)
        # loss2 = -F.mse_loss(input2, torch.tensor(0.5, requires_grad=False).cuda())
        # loss2 = -torch.std(input2, dim=-1).sum()
        # + F.mse_loss(torch.mean(input2, dim=-1), torch.tensor(0.2, requires_grad=False).cuda())

        # loss_std = 0
        # loss_avg = 0.
        # edge = edge_num + edge_num
        # edge_input2 = torch.split(input2, edge, dim=0)
        # for i in range(len(edge)):
        #     avg_e = 2 / (edge[i] * num_ops)
        #     loss_avg += 5 * F.mse_loss(torch.mean(edge_input2[i]), torch.tensor(avg_e, requires_grad=False).cuda())
        #     loss_std += -torch.std(edge_input2[i]).sum()
        # loss2 = loss_std + loss_avg

        # loss2 = torch.tensor([0.], device=input1.device)

        loss2 = - 0.2 * torch.std(input2)

        return loss1 + self.weight * loss2, loss1.item(), loss2.item()


class TriSeparateLoss(nn.modules.loss._Loss):
    """Separate the weight value between each operations using L1"""

    def __init__(self, loss1_fn, weight=0.1, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(TriSeparateLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.weight = weight
        self.loss1_fn = loss1_fn

    def forward(self, input1, target1, input2):
        loss1 = F.cross_entropy(input1, target1)
        loss2 = -F.l1_loss(input2, torch.tensor(0.5,
                           requires_grad=False).cuda())
        return loss1 + self.weight * loss2, loss1.item(), loss2.item()
