import json
import matplotlib.pyplot as plt
from braincog.model_zoo.darts.genotypes import Genotype, PRIMITIVES
import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from auto_augment import CIFAR10Policy
from braincog.model_zoo.darts.genotypes import PRIMITIVES

forward_edge_num = sum(1 for i in range(3) for n in range(2 + i))
backward_edge_num = sum(1 for i in range(3) for n in range(i))
num_ops = len(PRIMITIVES)
type_num = len(PRIMITIVES) // 2
# edge_num = [2, 3, 4]
edge_num = [2, 3, 4, 1, 2]


class AvgrageMeter(object):

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


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124] if args.dataset == 'cifar10' else [0.50707519, 0.48654887,
                                                                                         0.44091785]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768] if args.dataset == 'cifar10' else [0.26733428, 0.25643846,
                                                                                        0.27615049]

    normalize_transform = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]

    random_transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()]

    if args.auto_aug:
        random_transform += [CIFAR10Policy()]

    if args.cutout:
        cutout_transform = [Cutout(args.cutout_length)]
    else:
        cutout_transform = []

    train_transform = transforms.Compose(
        random_transform + normalize_transform + cutout_transform
    )

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def calc_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    t, h = divmod(h, 24)
    return {'day': t, 'hour': h, 'minute': m, 'second': int(s)}


def save_file(recoder, path='./', back_connection=False):
    size = (forward_edge_num +
            backward_edge_num if back_connection else forward_edge_num, num_ops)
    fig, axs = plt.subplots(*size, figsize=(36, 98))
    row = 0
    col = 0
    for (k, v) in recoder.items():
        axs[row, col].set_title(k)
        axs[row, col].plot(v, 'r+')
        if col == num_ops - 1:
            col = 0
            row += 1
        else:
            col += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, 'output.png'), bbox_inches='tight')
    plt.tight_layout()
    print('save history weight in {}'.format(os.path.join(path, 'output.png')))
    with open(os.path.join(path, 'history_weight.json'), 'w') as outf:
        json.dump(recoder, outf)
        print('save history weight in {}'.format(
            os.path.join(path, 'history_weight.json')))
