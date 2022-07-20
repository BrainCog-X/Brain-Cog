import os
import random
import math
import csv
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def setup_seed(seed):
    """
    为CPU，GPU，所有GPU，numpy，python设置随机数种子，并禁止hash随机化
    :param seed: seed value
    :return:
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)



def random_gradient(model: nn.Module, sigma: float):
    """
    为梯度添加噪声
    :param model: 模型
    :param sigma: 噪声方差
    :return:
    """
    for param in model.parameters():
        if param.grad is None:
            continue
        noise = torch.randn_like(param) * sigma
        param.grad = param.grad + noise


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """Compute the top1 and top5 accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # Return the k largest elements of the given input tensor
    # along a given dimension -> N * k
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mse(x, y):
    out = (x - y).pow(2).sum(-1, keepdim=True).mean()
    return out


def rand_ortho(shape, irange):
    A = - irange + 2 * irange * np.random.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return np.dot(U, np.dot(np.eye(U.shape[1], V.shape[0]), V))


def adjust_surrogate_coeff(epoch, tot_epochs):
    T_min, T_max = 1e-3, 1e1
    Kmin, Kmax = math.log(T_min) / math.log(10), math.log(T_max) / math.log(10)
    t = torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / tot_epochs * epoch)]).float().cuda()
    k = torch.tensor([1]).float().cuda()
    if k < 1:
        k = 1 / t
    return t, k


def save_feature_map(x, dir=''):
    for idx, layer in enumerate(x):
        layer = layer.cpu()
        for batch in range(layer.shape[0]):
            for channel in range(layer.shape[1]):
                fname = '{}_{}_{}_{}.jpg'.format(
                    idx, batch, channel, layer.shape[-1])
                fp = layer[batch, channel]
                plt.tight_layout()
                plt.axis('off')
                plt.imshow(fp, cmap='inferno')
                plt.savefig(os.path.join(dir, fname),
                            bbox_inches='tight', pad_inches=0)


def save_spike_info(fname, epoch, batch_idx, step, avg, var, spike, avg_per_step):
    """
    对spike-info格式进行调整, 便于保存
    :param fname: 输出文件名
    :param epoch: epoch
    :param batch_idx: batch index
    :param step: 仿真步长
    :param avg: 平均脉冲发放率
    :param var: 脉冲发放率的方差
    :param spike:
    :param avg_per_step:
    :return:
    """
    if not os.path.exists(fname):
        f = open(fname, mode='w', encoding='utf8', newline='')
        writer = csv.writer(f)
        head = ['epoch', 'batch', 'layer', 'avg', 'var']
        head.extend(['st_{}'.format(i) for i in range(step + 1)])  # spike times
        head.extend(['as_{}'.format(i) for i in range(step)])  # avg spike per time
        writer.writerow(head)

    else:
        f = open(fname, mode='a', encoding='utf8', newline='')
        writer = csv.writer(f)

    for layer in range(len(avg)):
        lst = [epoch, batch_idx, layer, avg[layer], var[layer]]
        lst.extend(spike[layer])
        lst.extend(avg_per_step[layer])
        lst = [str(x) for x in lst]
        writer.writerow(lst)


