import numpy as np
import torch
import numbers
import math


class IncConfusionMeter:
    """Maintains a confusion matrix for a given calssification problem.
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not
    """
    def __init__(self, k, increments, normalized=False):
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.increments = increments
        self.cum_increments = [0] + [sum(increments[:i + 1]) for i in range(len(increments))]
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.k**2)
        assert bincount_2d.size == self.k**2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        conf = self.conf.astype(np.float32)
        new_conf = np.zeros([len(self.increments), len(self.increments) + 2])
        for i in range(len(self.increments)):
            idxs = range(self.cum_increments[i], self.cum_increments[i + 1])
            new_conf[i, 0] = conf[idxs, idxs].sum()
            new_conf[i, 1] = conf[self.cum_increments[i]:self.cum_increments[i + 1],
                                  self.cum_increments[i]:self.cum_increments[i + 1]].sum() - new_conf[i, 0]
            for j in range(len(self.increments)):
                new_conf[i, j + 2] = conf[self.cum_increments[i]:self.cum_increments[i + 1],
                                          self.cum_increments[j]:self.cum_increments[j + 1]].sum()
        conf = new_conf
        if self.normalized:
            return conf / conf[:, 2:].sum(1).clip(min=1e-12)[:, None]
        else:
            return conf


class ClassErrorMeter:
    def __init__(self, topk=[1], accuracy=False):
        super(ClassErrorMeter, self).__init__()
        self.topk = np.sort(topk)
        self.accuracy = accuracy
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0

    def add(self, output, target):
        if isinstance(output, np.ndarray):
            output = torch.Tensor(output)
        if isinstance(target, np.ndarray):
            target = torch.Tensor(target)
        # if torch.is_tensor(output):
        #     output = output.cpu().squeeze().numpy()
        # if torch.is_tensor(target):
        #     target = target.cpu().squeeze().numpy()
        # elif isinstance(target, numbers.Number):
        #     target = np.asarray([target])
        # if np.ndim(output) == 1:
        #     output = output[np.newaxis]
        # else:
        #     assert np.ndim(output) == 2, \
        #         'wrong output size (1D or 2D expected)'
        #     assert np.ndim(target) == 1, \
        #         'target and output do not match'
        # assert target.shape[0] == output.shape[0], \
        #     'target and output do not match'
        topk = self.topk
        maxk = int(topk[-1])  # seems like Python3 wants int and not np.int64
        no = output.shape[0]

        pred = output.topk(maxk, 1, True, True)[1]
        correct = pred == target.unsqueeze(1).repeat(1, pred.shape[1])
        # pred = torch.from_numpy(output).topk(maxk, 1, True, True)[1].numpy()
        # correct = pred == target[:, np.newaxis].repeat(pred.shape[1], 1)

        for k in topk:
            self.sum[k] += no - correct[:, 0:k].sum()
        self.n += no

    def value(self, k=-1):
        if k != -1:
            assert k in self.sum.keys(), \
                'invalid k (this k was not provided at construction time)'
            if self.n == 0:
                return float('nan')
            if self.accuracy:
                return (1. - float(self.sum[k]) / self.n) * 100.0
            else:
                return float(self.sum[k]) / self.n * 100.0
        else:
            return [self.value(k_) for k_ in self.topk]


class AverageValueMeter:
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean, self.std = self.sum, np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = math.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan