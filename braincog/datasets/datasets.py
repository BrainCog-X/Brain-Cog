import os, warnings

import tonic
from tonic import DiskCachedDataset
 
import torch
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as datasets
from timm.data import ImageDataset, create_loader, Mixup, FastCollateMixup, AugMixDataset
from timm.data import create_transform

from torchvision import transforms
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from braincog.datasets.NOmniglot.nomniglot_full import NOmniglotfull
from braincog.datasets.NOmniglot.nomniglot_nw_ks import NOmniglotNWayKShot
from braincog.datasets.NOmniglot.nomniglot_pair import NOmniglotTrainSet, NOmniglotTestSet
from braincog.datasets.ESimagenet.ES_imagenet import ESImagenet_Dataset
from braincog.datasets.ESimagenet.reconstructed_ES_imagenet import ESImagenet2D_Dataset
from braincog.datasets.CUB2002011 import CUB2002011
from braincog.datasets.TinyImageNet import TinyImageNet
from braincog.datasets.StanfordDogs import StanfordDogs 

from .cut_mix import CutMix, EventMix, MixUp
from .rand_aug import *
from .utils import dvs_channel_check_expend, rescale

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


def build_transform(is_train, img_size):
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
    if resize_im:
        size = int((256 / 224) * img_size)
        t.append(
            # to maintain same ratio w.r.t. 224 images
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(img_size))

    t.append(transforms.ToTensor())
    if img_size > 32:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    else:
        t.append(transforms.Normalize(CIFAR10_DEFAULT_MEAN, CIFAR10_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, img_size, dataset, path, same_da=False):
    """
    构建带有增强策略的数据集
    :param is_train: 是否训练集
    :param img_size: 输出图像尺寸
    :param dataset: 数据集名称
    :param path: 数据集路径
    :param same_da: 为训练集使用测试集的增广方法
    :return: 增强后的数据集
    """
    transform = build_transform(False, img_size) if same_da else build_transform(is_train, img_size)

    if dataset == 'CIFAR10':
        dataset = datasets.CIFAR10(
            path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif dataset == 'CIFAR100':
        dataset = datasets.CIFAR100(
            path, train=is_train, transform=transform, download=True)
        nb_classes = 100
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


def get_mnist_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, **kwargs):
    """
    获取MNIST数据
    http://data.pymvpa.org/datasets/mnist/
    :param batch_size: batch size
    :param same_da: 为训练集使用测试集的增广方法
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    if 'root' in kwargs:root=kwargs["root"]
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
        root=root, train=True, transform=test_transform if same_da else train_transform, download=True)
    test_datasets = datasets.MNIST(
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


def get_fashion_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, **kwargs):
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
        root=root, train=True, transform=test_transform if same_da else train_transform, download=True)
    test_datasets = datasets.FashionMNIST(
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


def get_cifar10_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, **kwargs):
    """
    获取CIFAR10数据
     https://www.cs.toronto.edu/~kriz/cifar.html
    :param batch_size: batch size
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    train_datasets, _ = build_dataset(True, 32, 'CIFAR10', root, same_da)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR10', root, same_da)

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


def get_cifar100_data(batch_size, num_workers=8, same_data=False,root=DATA_DIR, *args, **kwargs):
    """
    获取CIFAR100数据
    https://www.cs.toronto.edu/~kriz/cifar.html
    :param batch_size: batch size
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    train_datasets, _ = build_dataset(True, 32, 'CIFAR100', root, same_data)
    test_datasets, _ = build_dataset(False, 32, 'CIFAR100', root, same_data)

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )
    return train_loader, test_loader, False, None

def get_TinyImageNet_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, *args, **kwargs):
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
    root=os.path.join(root, 'TinyImageNet')
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


def get_imnet_data(args, _logger, data_config, num_aug_splits,root=DATA_DIR, **kwargs):
    """
    获取ImageNet数据集
    http://arxiv.org/abs/1409.0575
    :param args: 其他的参数
    :param _logger: 日志路径
    :param data_config: 增强策略
    :param num_aug_splits: 不同增强策略的数量
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    train_dir = os.path.join(root, 'ILSVRC2012/train')
    if not os.path.exists(train_dir):
        _logger.error(
            'Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = ImageDataset(train_dir)
    # collate_fn = None
    # mixup_fn = None
    # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    # if mixup_active:
    #     mixup_args = dict(
    #         mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #         prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #         label_smoothing=args.smoothing, num_classes=args.num_classes)
    #     if args.prefetcher:
    #         # collate conflict (need to support deinterleaving in collate mixup)
    #         assert not num_aug_splits
    #         collate_fn = FastCollateMixup(**mixup_args)
    #     else:
    #         mixup_fn = Mixup(**mixup_args)

    # if num_aug_splits > 1:
    #     dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        # re_prob=args.reprob,
        # re_mode=args.remode,
        # re_count=args.recount,
        # re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        # vflip=arg,
        color_jitter=args.color_jitter,
        #auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        #collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader)
    eval_dir = os.path.join(root, 'ILSVRC2012/val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(root, 'ILSVRC2012/validation')
        if not os.path.isdir(eval_dir):
            _logger.error(
                'Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = ImageDataset(eval_dir)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    return loader_train, loader_eval, False, None


def get_dvsg_data(batch_size, step,root=DATA_DIR, **kwargs):
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

    train_dataset = tonic.datasets.DVSGesture(os.path.join(root, 'DVS/DVSGesture'),
                                              transform=train_transform, train=True)
    test_dataset = tonic.datasets.DVSGesture(os.path.join(root, 'DVS/DVSGesture'),
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
                                      cache_path=os.path.join(root, 'DVS/DVSGesture/train_cache_{}'.format(step)),
                                      transform=train_transform, num_copies=3)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(root, 'DVS/DVSGesture/test_cache_{}'.format(step)),
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


def get_dvsc10_data(batch_size, step,root=DATA_DIR, **kwargs):
    """
    获取DVS CIFAR10数据
    http://journal.frontiersin.org/article/10.3389/fnins.2017.00309/full
    :param batch_size: batch size
    :param step: 仿真步长
    :param kwargs:
    :return: (train loader, test loader, mixup_active, mixup_fn)
    """
    size = kwargs['size'] if 'size' in kwargs else 48
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    train_dataset = tonic.datasets.CIFAR10DVS(os.path.join(root, 'DVS/DVS_Cifar10'), transform=train_transform)
    test_dataset = tonic.datasets.CIFAR10DVS(os.path.join(root, 'DVS/DVS_Cifar10'), transform=test_transform)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        # lambda x: TemporalShift(x, .01),
        # lambda x: drop(x, 0.15),
        # lambda x: ShearX(x, 15),
        # lambda x: ShearY(x, 15),
        # lambda x: TranslateX(x, 0.225),
        # lambda x: TranslateY(x, 0.225),
        # lambda x: Rotate(x, 15),
        # lambda x: CutoutAbs(x, 0.25),
        # lambda x: CutoutTemporal(x, 0.25),
        # lambda x: GaussianBlur(x, 0.5),
        # lambda x: SaltAndPepperNoise(x, 0.1),
        # transforms.Normalize(DVSCIFAR10_MEAN_16, DVSCIFAR10_STD_16),
        transforms.RandomCrop(size, padding=size // 12),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    if 'rand_aug' in kwargs.keys():
        if kwargs['rand_aug'] is True:
            n = kwargs['randaug_n']
            m = kwargs['randaug_m']
            # print('randaug', m, n)
            train_transform.transforms.insert(2, RandAugment(m=m, n=n))

    # if 'temporal_flatten' in kwargs.keys():
    #     if kwargs['temporal_flatten'] is True:
    #         train_transform.transforms.insert(-1, lambda x: temporal_flatten(x))
    #         test_transform.transforms.insert(-1, lambda x: temporal_flatten(x))

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=os.path.join(root, 'DVS/DVS_Cifar10/train_cache_{}'.format(step)),
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(root, 'DVS/DVS_Cifar10/test_cache_{}'.format(step)),
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
        train_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_train),
        pin_memory=True, drop_last=True, num_workers=8
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices_test),
        pin_memory=True, drop_last=False, num_workers=2
    )

    return train_loader, test_loader, mixup_active, None


def get_NCALTECH101_data(batch_size, step,root=DATA_DIR, **kwargs):
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
        train_count += train_sample
        train_sample_weight.extend(
            [sample_weight] * train_sample
        )
        train_sample_weight.extend(
            [0.] * test_sample
        )
        train_sample_index.extend(
            list((range(idx_begin, idx_begin + train_sample)))
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

    train_dataset = tonic.datasets.NCALTECH101(os.path.join(root, 'DVS/NCALTECH101'), transform=train_transform)
    test_dataset = tonic.datasets.NCALTECH101(os.path.join(root, 'DVS/NCALTECH101'), transform=test_transform)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        # lambda x: print(x.shape),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        # transforms.RandomCrop(size, padding=size // 12),
        # transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15)
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
        # lambda x: temporal_flatten(x),
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
                                      cache_path=os.path.join(root, 'DVS/NCALTECH101/train_cache_{}'.format(step)),
                                      transform=train_transform, num_copies=3)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(root, 'DVS/NCALTECH101/test_cache_{}'.format(step)),
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


def get_NCARS_data(batch_size, step,root=DATA_DIR, **kwargs):
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

    train_dataset = tonic.datasets.NCARS(os.path.join(root, 'DVS/NCARS'), transform=train_transform, train=True)
    test_dataset = tonic.datasets.NCARS(os.path.join(root, 'DVS/NCARS'), transform=test_transform, train=False)

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
                                      cache_path=os.path.join(root, 'DVS/NCARS/train_cache_{}'.format(step)),
                                      transform=train_transform, num_copies=3)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=os.path.join(root, 'DVS/NCARS/test_cache_{}'.format(step)),
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


def get_nomni_data(batch_size, train_portion=1.,root=DATA_DIR, **kwargs):
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
        train_datasets = NOmniglotfull(root=os.path.join(root, 'DVS/NOmniglot'), train=True, frames_num=frames_num,
                                       data_type=data_type,
                                       transform=train_transform)
        test_datasets = NOmniglotfull(root=os.path.join(root, 'DVS/NOmniglot'), train=False, frames_num=frames_num,
                                      data_type=data_type,
                                      transform=test_transform)

    elif data_mode == "nkks":
        train_datasets = NOmniglotNWayKShot(os.path.join(root, 'DVS/NOmniglot'),
                                            n_way=kwargs["n_way"],
                                            k_shot=kwargs["k_shot"],
                                            k_query=kwargs["k_query"],
                                            train=True,
                                            frames_num=frames_num,
                                            data_type=data_type,
                                            transform=train_transform)
        test_datasets = NOmniglotNWayKShot(os.path.join(root, 'DVS/NOmniglot'),
                                           n_way=kwargs["n_way"],
                                           k_shot=kwargs["k_shot"],
                                           k_query=kwargs["k_query"],
                                           train=False,
                                           frames_num=frames_num,
                                           data_type=data_type,
                                           transform=test_transform)
    elif data_mode == "pair":
        train_datasets = NOmniglotTrainSet(root=os.path.join(root, 'DVS/NOmniglot'), use_frame=True,
                                           frames_num=frames_num, data_type=data_type,
                                           use_npz=False, resize=105)
        test_datasets = NOmniglotTestSet(root=os.path.join(root, 'DVS/NOmniglot'), time=2000, way=kwargs["n_way"],
                                         shot=kwargs["k_shot"], use_frame=True,
                                         frames_num=frames_num, data_type=data_type, use_npz=False, resize=105)

    else:
        pass

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, num_workers=12,
        pin_memory=True, drop_last=True, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size, num_workers=12,
        pin_memory=True, drop_last=False
    )
    return train_loader, test_loader, None, None


def get_esimnet_data(batch_size, step,root=DATA_DIR, **kwargs):
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
                                            data_set_path=os.path.join(root, 'DVS/ES-imagenet-0.18/extract/ES-imagenet-0.18/'),
                                            transform=train_transform)

        test_dataset = ESImagenet2D_Dataset(mode='test',
                                            data_set_path=os.path.join(root, 'DVS/ES-imagenet-0.18/extract/ES-imagenet-0.18/'),
                                            transform=test_transform)
    else:
        assert step == 8
        train_dataset = ESImagenet_Dataset(mode='train',
                                             data_set_path=os.path.join(root,
                                                                        'DVS/ES-imagenet-0.18/extract/ES-imagenet-0.18/'),
                                             transform=train_transform)

        test_dataset = ESImagenet_Dataset(mode='test',
                                            data_set_path=os.path.join(root,
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
        shuffle=False, sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=1,
        shuffle=False, sampler=test_sampler
    )

    return train_loader, test_loader, mixup_active, None


def get_CUB2002011_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, *args, **kwargs):
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
    root=os.path.join(root, 'CUB2002011')
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

def get_StanfordCars_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, *args, **kwargs):
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
    root=os.path.join(root, 'StanfordCars')
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

def get_StanfordDogs_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, *args, **kwargs):
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
    root=os.path.join(root, 'StanfordDogs')
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


def get_FGVCAircraft_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, *args, **kwargs):
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
    root=os.path.join(root, 'FGVCAircraft')
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


def get_Flowers102_data(batch_size, num_workers=8, same_da=False,root=DATA_DIR, *args, **kwargs):
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
    root=os.path.join(root, 'Flowers102')
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
