import os, warnings
import torchvision.datasets
try:
    import tonic
    from tonic import DiskCachedDataset
except:
    warnings.warn("tonic should be installed, 'pip install git+https://github.com/BrainCog-X/tonic_braincog.git'")
import torch
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as datasets
from timm.data import ImageDataset, create_loader, Mixup, FastCollateMixup, AugMixDataset
from timm.data import create_transform
from einops import rearrange, repeat
from torchvision import transforms
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from torch.utils.data import ConcatDataset
from braincog.datasets.NOmniglot.nomniglot_full import NOmniglotfull
from braincog.datasets.NOmniglot.nomniglot_nw_ks import NOmniglotNWayKShot
from braincog.datasets.NOmniglot.nomniglot_pair import NOmniglotTrainSet, NOmniglotTestSet
from braincog.datasets.ESimagenet.ES_imagenet import ESImagenet_Dataset
from braincog.datasets.ESimagenet.reconstructed_ES_imagenet import ESImagenet2D_Dataset
from braincog.datasets.CUB2002011 import CUB2002011
from braincog.datasets.TinyImageNet import TinyImageNet
from braincog.datasets.StanfordDogs import StanfordDogs 
from random import sample
from .cut_mix import CutMix, EventMix, MixUp
from .rand_aug import *
from .utils import dvs_channel_check_expend, rescale
from PIL import Image
import cv2
import math
DVSCIFAR10_MEAN_16 = [0.3290, 0.4507]
DVSCIFAR10_STD_16 = [1.8398, 1.6549]

DATA_DIR = '/data/datasets'

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

CIFAR10_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_DEFAULT_STD = (0.2023, 0.1994, 0.2010)


class TransferSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class Transfer_DataSet(torchvision.datasets.VisionDataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, mask):
        data = self.data[mask]
        label = self.label[mask]
        return data, label

    def __len__(self):
        return self.length


# 自定义HSV空间 transform
class ConvertHSV(object):
    """计算边缘梯度
    Args:
        None
    """

    def __init__(self):
        pass

    # transform 会调用该方法
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image, v channel.
        """
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return Image.fromarray(img.astype('uint8'))


def unpack_mix_param(args):
    mix_up = args['mix_up'] if 'mix_up' in args else False
    cut_mix = args['cut_mix'] if 'cut_mix' in args else False
    event_mix = args['event_mix'] if 'event_mix' in args else False
    beta = args['beta'] if 'beta' in args else 1.
    prob = args['prob'] if 'prob' in args else .5
    num = args['num'] if 'num' in args else 1
    num_classes = args['num_classes'] if 'num_classes' in args else 10
    noise = args['noise'] if 'noise' in args else 0.
    gaussian_n = args['gaussian_n'] if 'gaussian_n' in args else None
    return mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n


def build_transform(is_train, img_size, use_hsv=True):
    """
    构建数据增强, 适用于static data
    :param is_train: 是否训练集
    :param img_size: 输出的图像尺寸
    :return: 数据增强策略
    """
    resize_im = img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                img_size, padding=4)
        return transform

    t = []
    # if resize_im:
    #     size = int((256 / 224) * img_size)
    #     t.append(
    #         # to maintain same ratio w.r.t. 224 images
    #         transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
    #     )
    #     t.append(transforms.CenterCrop(img_size))

    # t.append(transforms.RandomAffine(degrees=0, translate=))
    # if Gradient:
    #     print("Used Gradient!")
    #     t.append(ComputeLaplacian())
        # t.append(ConvertHSV())
        # t.append(AddGaussianNoise())
    t.append(transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR))
    if use_hsv:
        print("Used V-channel!")
        t.append(ConvertHSV())
    t.append(transforms.ToTensor())
    # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, img_size, dataset, path, same_da=False, use_hsv=True):
    """
    构建带有增强策略的数据集
    :param is_train: 是否训练集
    :param img_size: 输出图像尺寸
    :param dataset: 数据集名称
    :param path: 数据集路径
    :param same_da: 为训练集使用测试集的增广方法
    : param use_hsv: 是否采用HSV
    :return: 增强后的数据集
    """
    # transform = build_transform(False, img_size) if same_da else build_transform(is_train, img_size)
    transform = build_transform(False, img_size, use_hsv) if same_da else build_transform(False, img_size, use_hsv)
    if dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(
            path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif dataset == 'CIFAR100':
        dataset = datasets.CIFAR100(
            path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif dataset == 'CALTECH101':
        dataset = datasets.Caltech101(
            path, transform=transform, download=True
        )
        nb_classes = 101
    else:
        raise NotImplementedError

    return dataset, nb_classes


class MNISTData(object):
    """
    Load MNIST datesets.
    """

    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 train_trans: Sequence[torch.nn.Module] = None,
                 test_trans: Sequence[torch.nn.Module] = None,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 shuffle: bool = True,
                 ) -> None:
        self._data_path = data_path
        self._batch_size = batch_size
        self._pin_memory = pin_memory
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._train_transform = transforms.Compose(train_trans) if train_trans else None
        self._test_transform = transforms.Compose(test_trans) if test_trans else None

    def get_data_loaders(self):
        print('Batch size: ', self._batch_size)
        train_datasets = datasets.MNIST(root=self._data_path, train=True, transform=self._train_transform, download=True)
        test_datasets = datasets.MNIST(root=self._data_path, train=False, transform=self._test_transform, download=True)
        train_loader = torch.utils.data.DataLoader(
            train_datasets, batch_size=self._batch_size,
            pin_memory=self._pin_memory, drop_last=self._drop_last, shuffle=self._shuffle
        )
        test_loader = torch.utils.data.DataLoader(
            test_datasets, batch_size=self._batch_size,
            pin_memory=self._pin_memory, drop_last=False
        )
        return train_loader, test_loader

    def get_standard_data(self):
        MNIST_MEAN = 0.1307
        MNIST_STD = 0.3081
        self._train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
        self._test_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
        return self.get_data_loaders()


def get_mnist_data(batch_size, num_workers=8, same_da=False, **kwargs):
    """s
    获取MNIST数据
    http://data.pymvpa.org/datasets/mnist/
    :param batch_size: batch size
    :param same_da: 为训练集使用测试集的增广方法
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    if 'skip_norm' in kwargs and kwargs['skip_norm'] is True:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(rescale)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(rescale)
        ])
    else:
        train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                              # transforms.RandomRotation(10),
                                              transforms.ToTensor(),
                                              transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])

    train_datasets = datasets.MNIST(
        root=DATA_DIR, train=True, transform=test_transform if same_da else train_transform, download=True)
    test_datasets = datasets.MNIST(
        root=DATA_DIR, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None


def get_fashion_data(batch_size, num_workers=8, same_da=False, **kwargs):
    """
    获取fashion MNIST数据
    http://arxiv.org/abs/1708.07747
    :param batch_size: batch size
    :param same_da: 为训练集使用测试集的增广方法
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    train_transform = transforms.Compose([transforms.RandomCrop(28, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_datasets = datasets.FashionMNIST(
        root=DATA_DIR, train=True, transform=test_transform if same_da else train_transform, download=True)
    test_datasets = datasets.FashionMNIST(
        root=DATA_DIR, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None


def get_cifar10_data(batch_size, num_workers=8, same_da=False, **kwargs):
    """
    获取CIFAR10数据
     https://www.cs.toronto.edu/~kriz/cifar.html
    :param batch_size: batch size
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    use_hsv = not kwargs['no_use_hsv'] if 'no_use_hsv' in kwargs else True
    train_datasets, _ = build_dataset(True, 32, 'CIFAR10', DATA_DIR, same_da, False)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR10', DATA_DIR, same_da, False)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False,
        num_workers=num_workers
    )
    return train_loader, test_loader, None, None


def get_cifar100_data(batch_size, num_workers=8, same_data=False, *args, **kwargs):
    """
    获取CIFAR100数据
    https://www.cs.toronto.edu/~kriz/cifar.html
    :param batch_size: batch size
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    train_datasets, _ = build_dataset(True, 32, 'CIFAR100', DATA_DIR, same_data)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR100', DATA_DIR, same_data)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )
    return train_loader, test_loader, False, None


def get_transfer_cifar10_data(batch_size, num_workers=8, same_da=False, **kwargs):
    use_hsv = not kwargs['no_use_hsv'] if 'no_use_hsv' in kwargs else True
    train_datasets, _ = build_dataset(True, 48, 'CIFAR10', DATA_DIR, same_da, use_hsv)  # 原来是48
    test_datasets, _ = build_dataset(False, 48, 'CIFAR10', DATA_DIR, same_da, use_hsv)

    concat_dataset = ConcatDataset([train_datasets, test_datasets])  # concat dataset

    img_index = [[] for i in range(10)]
    label_index = [0] * 60000
    for idx, (img, label) in enumerate(concat_dataset):
        img_index[label].append(img)
    for i in range(10):
        img_index[i] = torch.stack(img_index[i], 0)
        label_index[i * 6000:2 * i * 6000] = [i] * 6000
    source_datasets = Transfer_DataSet(data=rearrange(torch.stack(img_index, dim=0), 'l b c w h -> (l b) c w h'),
                                       label=label_index)

    source_loader = torch.utils.data.DataLoader(
        source_datasets, batch_size=60000,
        sampler=TransferSampler(torch.arange(0, 60000).tolist()),
        pin_memory=True, drop_last=False, num_workers=16
    )
    return source_loader, None, None, None


def get_combined_cifar10_data(batch_size, num_workers=8, same_da=False, **kwargs):
    use_hsv = not kwargs['no_use_hsv'] if 'no_use_hsv' in kwargs else True
    train_datasets, _ = build_dataset(True, 48, 'CIFAR10', DATA_DIR, same_da, use_hsv)
    test_datasets, _ = build_dataset(False, 48, 'CIFAR10', DATA_DIR, same_da, use_hsv)

    concat_dataset = ConcatDataset([train_datasets, test_datasets])  # concat dataset

    source_loader = torch.utils.data.DataLoader(
        concat_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=8, shuffle=True
    )
    return source_loader, None, None, None


def get_transfer_CALTECH101_data(batch_size, num_workers=8, same_da=False, **kwargs):
    """
    获取NCaltech101数据
    http://journal.frontiersin.org/Article/10.3389/fnins.2015.00437/abstract
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    use_hsv = not kwargs['no_use_hsv'] if 'no_use_hsv' in kwargs else True
    datasets, _ = build_dataset(False, 48, 'CALTECH101', DATA_DIR, same_da, use_hsv)
    dataset_length = 8299

    train_loader = torch.utils.data.DataLoader(
        datasets, batch_size=10000,
        sampler=TransferSampler(torch.arange(0, dataset_length).tolist()),
        pin_memory=True, drop_last=False, num_workers=4
    )

    return train_loader, None, None, None


def get_combined_CALTECH101_data(batch_size, num_workers=8, same_da=False, **kwargs):
    """
    获取NCaltech101数据
    http://journal.frontiersin.org/Article/10.3389/fnins.2015.00437/abstract
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    use_hsv = not kwargs['no_use_hsv'] if 'no_use_hsv' in kwargs else True
    datasets, _ = build_dataset(False, 48, 'CALTECH101', DATA_DIR, same_da, use_hsv)
    dataset_length = 8299

    train_loader = torch.utils.data.DataLoader(
        datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False,
        num_workers=4, shuffle=True
    )

    return train_loader, None, None, None


def get_TinyImageNet_data(batch_size, num_workers=8, same_da=False, *args, **kwargs):
    size=kwargs["size"] if "size" in kwargs else 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(size*8//7),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    root=os.path.join(DATA_DIR, 'TinyImageNet')
    train_datasets = TinyImageNet(
        root=root, split="train", transform=test_transform if same_da else train_transform, download=True)
    test_datasets = TinyImageNet(
        root=root, split="val", transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None


def get_transfer_imnet_data(args, _logger, data_config, num_aug_splits, **kwargs):
    '''
    load imagenet 2012
    we use images in train/ for training, and use images in val/ for testing
    https://github.com/pytorch/examples/tree/master/imagenet
    '''
    IMAGENET_PATH = '/data/datasets/ILSVRC2012/'
    traindir = os.path.join(IMAGENET_PATH, 'train')
    valdir = os.path.join(IMAGENET_PATH, 'val')
    batch_size = kwargs['batch_size']

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            ConvertHSV(),
            transforms.ToTensor()]))

    # val_dataset = datasets.ImageFolder(
    #     valdir,
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         ConvertHSV(),
    #         transforms.ToTensor()]))

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=4, pin_memory=True, sampler=TransferSampler([0, 1300, 2599, 2600]))
    #
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=4, pin_memory=True)
    return train_dataset, None, None, None


def get_dvsg_data(batch_size, step, **kwargs):
    """
    获取DVS Gesture数据
    DOI: 10.1109/CVPR.2017.781
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    size = kwargs['size'] if 'size' in kwargs else 48

    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
    ])

    train_dataset = tonic.datasets.DVSGesture(os.path.join(DATA_DIR, 'DVS/DVSGesture'),
                                              transform=train_transform, train=True)
    test_dataset = tonic.datasets.DVSGesture(os.path.join(DATA_DIR, 'DVS/DVSGesture'),
                                             transform=test_transform, train=False)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: dvs_channel_check_expend(x),
        transforms.RandomCrop(size, padding=size // 12),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15)
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: dvs_channel_check_expend(x),
    ])
    if 'rand_aug' in kwargs.keys():
        if kwargs['rand_aug'] is True:
            n = kwargs['randaug_n']
            m = kwargs['randaug_m']
            train_transform.transforms.insert(2, RandAugment(m=m, n=n))

    # if 'temporal_flatten' in kwargs.keys():
    #     if kwargs['temporal_flatten'] is True:
    #         train_transform.transforms.insert(-1, lambda x: temporal_flatten(x))
    #         test_transform.transforms.insert(-1, lambda x: temporal_flatten(x))

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(DATA_DIR, 'DVS/DVSGesture/train_cache_{}'.format(step)),
                                      transform=train_transform, num_copies=3)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(DATA_DIR, 'DVS/DVSGesture/test_cache_{}'.format(step)),
                                     transform=test_transform, num_copies=3)

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        train_dataset = CutMix(train_dataset,
                               beta=beta,
                               prob=prob,
                               num_mix=num,
                               num_class=num_classes,
                               noise=noise)

    if event_mix:
        train_dataset = EventMix(train_dataset,
                                 beta=beta,
                                 prob=prob,
                                 num_mix=num,
                                 num_class=num_classes,
                                 noise=noise,
                                 gaussian_n=gaussian_n)
    if mix_up:
        train_dataset = MixUp(train_dataset,
                              beta=beta,
                              prob=prob,
                              num_mix=num,
                              num_class=num_classes,
                              noise=noise)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=True, num_workers=8,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=2,
        shuffle=False,
    )

    return train_loader, test_loader, mixup_active, None


def get_dvsc10_data(batch_size, step, dvs_da=False, **kwargs):
    """
    获取DVS CIFAR10数据
    http://journal.frontiersin.org/article/10.3389/fnins.2017.00309/full
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    size = kwargs['size'] if 'size' in kwargs else 48
    snr = kwargs['snr'] if 'snr' in kwargs else 0
    train_data_ratio = kwargs['train_data_ratio'] if 'train_data_ratio' in kwargs else 1.0
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    train_dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=train_transform)
    test_dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=test_transform)

    if dvs_da is True:
        print("use dvs_da")
        if snr > 0:
            train_transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: x + torch.randn(x.shape) * math.sqrt(torch.mean(torch.pow(x, 2)) / math.pow(10, snr / 10)),
                transforms.RandomCrop(size, padding=size // 12),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ])
        else:
            train_transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                transforms.RandomCrop(size, padding=size // 12),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ])
    else:
        train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])   # 这里lambda返回的是地址, 注意不要用List复用.

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(DATA_DIR, 'DVS/DVS_Cifar10/train_cache_{}'.format(step)),
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(DATA_DIR, 'DVS/DVS_Cifar10/test_cache_{}'.format(step)),
                                     transform=test_transform)

    num_train = len(train_dataset)
    num_per_cls = num_train // 10
    indices_train, indices_test = [], []
    portion = kwargs['portion'] if 'portion' in kwargs else .9
    for i in range(10):
        indices_train.extend(
            sample(list(range(i * num_per_cls, round(i * num_per_cls + num_per_cls * portion))), int(num_per_cls * portion * train_data_ratio)))
        indices_test.extend(
            list(range(round(i * num_per_cls + num_per_cls * portion), (i + 1) * num_per_cls)))

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        # print('cut_mix', beta, prob, num, num_classes)
        train_dataset = CutMix(train_dataset,
                               beta=beta,
                               prob=prob,
                               num_mix=num,
                               num_class=num_classes,
                               indices=indices_train,
                               noise=noise)

    if event_mix:
        train_dataset = EventMix(train_dataset,
                                 beta=beta,
                                 prob=prob,
                                 num_mix=num,
                                 num_class=num_classes,
                                 indices=indices_train,
                                 noise=noise,
                                 gaussian_n=gaussian_n)

    if mix_up:
        train_dataset = MixUp(train_dataset,
                              beta=beta,
                              prob=prob,
                              num_mix=num,
                              num_class=num_classes,
                              indices=indices_train,
                              noise=noise)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
        pin_memory=True, drop_last=False, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        pin_memory=True, drop_last=False, num_workers=2
    )

    return train_loader, test_loader, mixup_active, None



def get_transfer_dvsc10_data(batch_size, step, dvs_da=False, **kwargs):
    """
    获取DVS CIFAR10数据
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    size = kwargs['size'] if 'size' in kwargs else 48
    snr = kwargs['snr'] if 'snr' in kwargs else 0
    train_data_ratio = kwargs['train_data_ratio'] if 'train_data_ratio' in kwargs else 1.0
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    train_dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=train_transform)
    test_dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=test_transform)

    train_transform = transforms.Compose([
    lambda x: torch.tensor(x, dtype=torch.float),
    lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),])


    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])   # 这里lambda返回的是地址, 注意不要用List复用.

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(DATA_DIR, 'DVS/DVS_Cifar10/train_cache_{}'.format(step)),
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(DATA_DIR, 'DVS/DVS_Cifar10/test_cache_{}'.format(step)),
                                     transform=test_transform)

    num_train = len(train_dataset)
    num_per_cls = num_train // 10
    indices_train, indices_test = [], []
    portion = kwargs['portion'] if 'portion' in kwargs else .9
    for i in range(10):
        indices_train.extend(
            list(range(i * num_per_cls, round(i * num_per_cls + num_per_cls * portion))))
        indices_test.extend(
            list(range(round(i * num_per_cls + num_per_cls * portion), (i + 1) * num_per_cls)))

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        # print('cut_mix', beta, prob, num, num_classes)
        train_dataset = CutMix(train_dataset,
                               beta=beta,
                               prob=prob,
                               num_mix=num,
                               num_class=num_classes,
                               indices=indices_train,
                               noise=noise)

    if event_mix:
        train_dataset = EventMix(train_dataset,
                                 beta=beta,
                                 prob=prob,
                                 num_mix=num,
                                 num_class=num_classes,
                                 indices=indices_train,
                                 noise=noise,
                                 gaussian_n=gaussian_n)

    if mix_up:
        train_dataset = MixUp(train_dataset,
                              beta=beta,
                              prob=prob,
                              num_mix=num,
                              num_class=num_classes,
                              indices=indices_train,
                              noise=noise)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(indices_train),
        sampler=TransferSampler(indices_train),
        pin_memory=True, drop_last=True, num_workers=8
    )

    return train_loader, None, mixup_active, None


def get_NCALTECH101_data(batch_size, step, dvs_da=False, **kwargs):
    """
    获取NCaltech101数据
    http://journal.frontiersin.org/Article/10.3389/fnins.2015.00437/abstract
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    sensor_size = tonic.datasets.NCALTECH101.sensor_size
    cls_count = tonic.datasets.NCALTECH101.cls_count
    dataset_length = tonic.datasets.NCALTECH101.length
    portion = kwargs['portion'] if 'portion' in kwargs else .9
    size = kwargs['size'] if 'size' in kwargs else 48
    snr = kwargs['snr'] if 'snr' in kwargs else 0
    train_data_ratio = kwargs['train_data_ratio'] if 'train_data_ratio' in kwargs else 1.0
    # print('portion', portion)
    train_sample_weight = []
    train_sample_index = []
    train_count = 0
    test_sample_index = []
    idx_begin = 0
    for count in cls_count:
        sample_weight = dataset_length / count
        train_sample = round(portion * count)
        test_sample = count - train_sample
        train_count += int(train_sample * train_data_ratio)
        train_sample_weight.extend(
            [sample_weight] * int(train_sample * train_data_ratio)
        )
        train_sample_weight.extend(
            [0.] * (train_sample - int(train_sample * train_data_ratio))
        )
        train_sample_weight.extend(
            [0.] * test_sample
        )
        train_sample_index.extend(
            sample(list(range(idx_begin, idx_begin + train_sample)), int(train_sample * train_data_ratio))
        )
        test_sample_index.extend(
            list(range(idx_begin + train_sample, idx_begin + train_sample + test_sample))
        )
        idx_begin += count

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sample_weight, train_count)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_sample_index)

    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])

    train_dataset = tonic.datasets.NCALTECH101(os.path.join(DATA_DIR, 'DVS/NCALTECH101'), transform=train_transform)
    test_dataset = tonic.datasets.NCALTECH101(os.path.join(DATA_DIR, 'DVS/NCALTECH101'), transform=test_transform)

    if dvs_da is True:
        print("use dvs_da")
        train_transform = transforms.Compose([
            lambda x: torch.tensor(x, dtype=torch.float),
            lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
            transforms.RandomCrop(size, padding=size // 12),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])
    else:
        if snr > 0:
            train_transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                lambda x: x + torch.randn(x.shape) * math.sqrt(torch.mean(torch.pow(x, 2)) / math.pow(10, snr / 10)),
            ])
        else:
            train_transform = transforms.Compose([
                lambda x: torch.tensor(x, dtype=torch.float),
                lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                transforms.RandomCrop(size, padding=size // 12),
            ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])  # 这里lambda返回的是地址, 注意不要用List复用.

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(DATA_DIR, 'DVS/NCALTECH101/train_cache_{}'.format(step)),
                                      transform=train_transform, num_copies=3)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(DATA_DIR, 'DVS/NCALTECH101/test_cache_{}'.format(step)),
                                     transform=test_transform, num_copies=3)

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        train_dataset = CutMix(train_dataset,
                               beta=beta,
                               prob=prob,
                               num_mix=num,
                               num_class=num_classes,
                               indices=train_sample_index,
                               noise=noise)

    if event_mix:
        train_dataset = EventMix(train_dataset,
                                 beta=beta,
                                 prob=prob,
                                 num_mix=num,
                                 num_class=num_classes,
                                 indices=train_sample_index,
                                 noise=noise,
                                 gaussian_n=gaussian_n)
    if mix_up:
        train_dataset = MixUp(train_dataset,
                              beta=beta,
                              prob=prob,
                              num_mix=num,
                              num_class=num_classes,
                              indices=train_sample_index,
                              noise=noise)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True, drop_last=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=True, drop_last=False, num_workers=2
    )

    return train_loader, test_loader, mixup_active, None


def get_transfer_NCALTECH101_data(batch_size, step, dvs_da=False, **kwargs):
    """
    获取NCaltech101数据
    http://journal.frontiersin.org/Article/10.3389/fnins.2015.00437/abstract
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    sensor_size = tonic.datasets.NCALTECH101.sensor_size
    cls_count = tonic.datasets.NCALTECH101.cls_count
    dataset_length = tonic.datasets.NCALTECH101.length
    portion = kwargs['portion'] if 'portion' in kwargs else .9
    size = kwargs['size'] if 'size' in kwargs else 48
    snr = kwargs['snr'] if 'snr' in kwargs else 0
    train_data_ratio = kwargs['train_data_ratio'] if 'train_data_ratio' in kwargs else 1.0
    # print('portion', portion)
    train_sample_weight = []
    train_sample_index = []
    train_count = 0
    test_sample_index = []
    idx_begin = 0
    for count in cls_count:
        sample_weight = dataset_length / count
        train_sample = round(portion * count)
        test_sample = count - train_sample
        train_count += int(train_sample * train_data_ratio)
        train_sample_weight.extend(
            [sample_weight] * int(train_sample * train_data_ratio)
        )
        train_sample_weight.extend(
            [0.] * (train_sample - int(train_sample * train_data_ratio))
        )
        train_sample_weight.extend(
            [0.] * test_sample
        )
        train_sample_index.extend(
            sample(list(range(idx_begin, idx_begin + train_sample)), int(train_sample * train_data_ratio))
        )
        test_sample_index.extend(
            list(range(idx_begin + train_sample, idx_begin + train_sample + test_sample))
        )
        idx_begin += count

    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sample_weight, train_count)
    test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_sample_index)

    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])

    train_dataset = tonic.datasets.NCALTECH101(os.path.join(DATA_DIR, 'DVS/NCALTECH101'), transform=train_transform)
    test_dataset = tonic.datasets.NCALTECH101(os.path.join(DATA_DIR, 'DVS/NCALTECH101'), transform=test_transform)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])  # 这里lambda返回的是地址, 注意不要用List复用.

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(DATA_DIR, 'DVS/NCALTECH101/train_cache_{}'.format(step)),
                                      transform=train_transform, num_copies=3)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(DATA_DIR, 'DVS/NCALTECH101/test_cache_{}'.format(step)),
                                     transform=test_transform, num_copies=3)

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_sample_index),
        sampler=TransferSampler(train_sample_index),
        pin_memory=True, drop_last=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=True, drop_last=False, num_workers=2
    )

    return train_loader, None, None, None


def get_NCARS_data(batch_size, step, **kwargs):
    """
    获取N-Cars数据
    https://ieeexplore.ieee.org/document/8578284/
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    sensor_size = tonic.datasets.NCARS.sensor_size
    size = kwargs['size'] if 'size' in kwargs else 48

    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=None, n_time_bins=step),
    ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=None, n_time_bins=step),
    ])

    train_dataset = tonic.datasets.NCARS(os.path.join(DATA_DIR, 'DVS/NCARS'), transform=train_transform, train=True)
    test_dataset = tonic.datasets.NCARS(os.path.join(DATA_DIR, 'DVS/NCARS'), transform=test_transform, train=False)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: dvs_channel_check_expend(x),
        transforms.RandomCrop(size, padding=size // 12),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        lambda x: dvs_channel_check_expend(x),
    ])
    if 'rand_aug' in kwargs.keys():
        if kwargs['rand_aug'] is True:
            n = kwargs['randaug_n']
            m = kwargs['randaug_m']
            train_transform.transforms.insert(2, RandAugment(m=m, n=n))

    # if 'temporal_flatten' in kwargs.keys():
    #     if kwargs['temporal_flatten'] is True:
    #         train_transform.transforms.insert(-1, lambda x: temporal_flatten(x))
    #         test_transform.transforms.insert(-1, lambda x: temporal_flatten(x))

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(DATA_DIR, 'DVS/NCARS/train_cache_{}'.format(step)),
                                      transform=train_transform, num_copies=3)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(DATA_DIR, 'DVS/NCARS/test_cache_{}'.format(step)),
                                     transform=test_transform, num_copies=3)

    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        train_dataset = CutMix(train_dataset,
                               beta=beta,
                               prob=prob,
                               num_mix=num,
                               num_class=num_classes,
                               noise=noise)

    if event_mix:
        train_dataset = EventMix(train_dataset,
                                 beta=beta,
                                 prob=prob,
                                 num_mix=num,
                                 num_class=num_classes,
                                 noise=noise,
                                 gaussian_n=gaussian_n)
    if mix_up:
        train_dataset = MixUp(train_dataset,
                              beta=beta,
                              prob=prob,
                              num_mix=num,
                              num_class=num_classes,
                              noise=noise)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=True, num_workers=8,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=2,
        shuffle=False,
    )

    return train_loader, test_loader, mixup_active, None


def get_nomni_data(batch_size, train_portion=1., **kwargs):
    """
    获取N-Omniglot数据
    :param batch_size:batch的大小
    :param data_mode:一共full nkks pair三种模式
    :param frames_num:一个样本帧的个数
    :param data_type:event frequency两种模式
    """
    data_mode = kwargs["data_mode"] if "data_mode" in kwargs else "full"
    frames_num = kwargs["frames_num"] if "frames_num" in kwargs else 4
    data_type = kwargs["data_type"] if "data_type" in kwargs else "event"

    train_transform = transforms.Compose([
        transforms.Resize((28, 28))])
    test_transform = transforms.Compose([
        transforms.Resize((28, 28))])
    if data_mode == "full":
        train_datasets = NOmniglotfull(root=os.path.join(DATA_DIR, 'DVS/NOmniglot'), train=True, frames_num=frames_num,
                                       data_type=data_type,
                                       transform=train_transform, use_npz=True)
        test_datasets = NOmniglotfull(root=os.path.join(DATA_DIR, 'DVS/NOmniglot'), train=False, frames_num=frames_num,
                                      data_type=data_type,
                                      transform=test_transform, use_npz=True)

    elif data_mode == "nkks":
        train_datasets = NOmniglotNWayKShot(os.path.join(DATA_DIR, 'DVS/NOmniglot'),
                                            n_way=kwargs["n_way"],
                                            k_shot=kwargs["k_shot"],
                                            k_query=kwargs["k_query"],
                                            train=True,
                                            frames_num=frames_num,
                                            data_type=data_type,
                                            transform=train_transform)
        test_datasets = NOmniglotNWayKShot(os.path.join(DATA_DIR, 'DVS/NOmniglot'),
                                           n_way=kwargs["n_way"],
                                           k_shot=kwargs["k_shot"],
                                           k_query=kwargs["k_query"],
                                           train=False,
                                           frames_num=frames_num,
                                           data_type=data_type,
                                           transform=test_transform)
    elif data_mode == "pair":
        train_datasets = NOmniglotTrainSet(root=os.path.join(DATA_DIR, 'DVS/NOmniglot'), use_frame=True,
                                           frames_num=frames_num, data_type=data_type,
                                           use_npz=False, resize=105)
        test_datasets = NOmniglotTestSet(root=os.path.join(DATA_DIR, 'DVS/NOmniglot'), time=2000, way=kwargs["n_way"],
                                         shot=kwargs["k_shot"], use_frame=True,
                                         frames_num=frames_num, data_type=data_type, use_npz=False, resize=105)

    else:
        pass

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, num_workers=4,
        pin_memory=True, drop_last=True, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, num_workers=4,
        pin_memory=True, drop_last=False
    )
    return train_loader, test_loader, None, None



def get_transfer_omni_data(batch_size, train_portion=1., **kwargs):
    """
    获取Omniglot数据
    :param batch_size:batch的大小
    :param data_mode:一共full nkks pair三种模式
    :param frames_num:一个样本帧的个数
    :param data_type:event frequency两种模式
    """

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()])

    train_dataset = datasets.Omniglot(
        root="/data/datasets/", background=True, download=True, transform=transform
    )
    test_dataset = datasets.Omniglot(
        root="/data/datasets/", background=False, download=True, transform=transform
    )
    dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    dataset_length = len(dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=35000, num_workers=12,
        pin_memory=True, drop_last=False,
        sampler=TransferSampler(torch.arange(0, dataset_length).tolist())
    )

    return train_loader, None, None, None


def get_esimnet_data(batch_size, step, **kwargs):
    """
    获取ES imagenet数据
    DOI: 10.3389/fnins.2021.726582
    :param batch_size: batch size
    :param step: 仿真步长，固定为8
    :param reconstruct: 重构则时间步为1, 否则为8
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    :note: 没有自动下载, 下载及md5请参考spikingjelly, sampler默认为DistributedSampler
    """

    reconstruct = kwargs["reconstruct"] if "reconstruct" in kwargs else False

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])
    test_transform = transforms.Compose([
        lambda x: dvs_channel_check_expend(x),
    ])

    if reconstruct:
        assert step == 1
        train_dataset = ESImagenet2D_Dataset(mode='train',
                                            data_set_path=os.path.join(DATA_DIR, 'DVS/ES-imagenet-0.18/extract/ES-imagenet-0.18/'),
                                            transform=train_transform)

        test_dataset = ESImagenet2D_Dataset(mode='test',
                                            data_set_path=os.path.join(DATA_DIR, 'DVS/ES-imagenet-0.18/extract/ES-imagenet-0.18/'),
                                            transform=test_transform)
    else:
        assert step == 8
        train_dataset = ESImagenet_Dataset(mode='train',
                                             data_set_path=os.path.join(DATA_DIR,
                                                                        'DVS/ES-imagenet-0.18/extract/ES-imagenet-0.18/'),
                                             transform=train_transform)

        test_dataset = ESImagenet_Dataset(mode='test',
                                            data_set_path=os.path.join(DATA_DIR,
                                                                       'DVS/ES-imagenet-0.18/extract/ES-imagenet-0.18/'),
                                            transform=test_transform)


    mix_up, cut_mix, event_mix, beta, prob, num, num_classes, noise, gaussian_n = unpack_mix_param(kwargs)
    mixup_active = cut_mix | event_mix | mix_up

    if cut_mix:
        train_dataset = CutMix(train_dataset,
                               beta=beta,
                               prob=prob,
                               num_mix=num,
                               num_class=num_classes,
                               noise=noise)

    if event_mix:
        train_dataset = EventMix(train_dataset,
                                 beta=beta,
                                 prob=prob,
                                 num_mix=num,
                                 num_class=num_classes,
                                 noise=noise,
                                 gaussian_n=gaussian_n)
    if mix_up:
        train_dataset = MixUp(train_dataset,
                              beta=beta,
                              prob=prob,
                              num_mix=num,
                              num_class=num_classes,
                              noise=noise)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=True, num_workers=8,
        sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=8,
        sampler=test_sampler
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size,
    #     pin_memory=True, drop_last=True, num_workers=8,
    #     shuffle=True
    # )
    #
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size,
    #     pin_memory=True, drop_last=False, num_workers=1,
    #     shuffle=False
    # )

    return train_loader, test_loader, mixup_active, None


def get_CUB2002011_data(batch_size, num_workers=8, same_da=False, *args, **kwargs):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    root=os.path.join(DATA_DIR, 'CUB2002011')
    train_datasets = CUB2002011(
        root=root, train=True, transform=test_transform if same_da else train_transform, download=True)
    test_datasets = CUB2002011(
        root=root, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None

def get_StanfordCars_data(batch_size, num_workers=8, same_da=False, *args, **kwargs):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    root=os.path.join(DATA_DIR, 'StanfordCars')
    train_datasets = datasets.StanfordCars(
        root=root, split ="train", transform=test_transform if same_da else train_transform, download=True)
    test_datasets = datasets.StanfordCars(
        root=root, split ="test", transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None

def get_StanfordDogs_data(batch_size, num_workers=8, same_da=False, *args, **kwargs):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    root=os.path.join(DATA_DIR, 'StanfordDogs')
    train_datasets = StanfordDogs(
        root=root, train=True, transform=test_transform if same_da else train_transform, download=True)
    test_datasets = StanfordDogs(
        root=root, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None


def get_FGVCAircraft_data(batch_size, num_workers=8, same_da=False, *args, **kwargs):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    root=os.path.join(DATA_DIR, 'FGVCAircraft')
    train_datasets = datasets.FGVCAircraft(
        root=root, split="train", transform=test_transform if same_da else train_transform, download=True)
    test_datasets = datasets.FGVCAircraft(
        root=root, split="test", transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None


def get_Flowers102_data(batch_size, num_workers=8, same_da=False, *args, **kwargs):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    root=os.path.join(DATA_DIR, 'Flowers102')
    train_datasets = datasets.Flowers102(
        root=root, split="train", transform=test_transform if same_da else train_transform, download=True)
    test_datasets = datasets.Flowers102(
        root=root, split="test", transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None