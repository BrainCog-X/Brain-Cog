from .criterions import UnilateralMse, MixLoss
from .visualization import plot_tsne, plot_tsne_3d, plot_confusion_matrix
from torch.autograd import Variable
import torch

__all__ = [
    'UnilateralMse', 'MixLoss',
    'plot_tsne', 'plot_tsne_3d', 'plot_confusion_matrix', 'drop_path'
]


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x
