import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
import tonic
from tonic import DiskCachedDataset
import torch.nn.functional as F
import os


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
cifar100_mean = [0.5071, 0.4865, 0.4409]
cifar100_std = [0.2673, 0.2563, 0.2761]
DVSCIFAR10_MEAN_16 = [0.3290, 0.4507]
DVSCIFAR10_STD_16 = [1.8398, 1.6549]

DATA_DIR = '/data/datasets'


class CustomDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        x, y = self.dataset[self.indices[item]]
        return x, y


def load_static_data(data_root, batch_size, dataset):
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV)])

        train_data = datasets.CIFAR10(data_root, train=True, transform=transform_train, download=True)
        test_data = datasets.CIFAR10(data_root, train=False, transform=transform_test, download=True)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
        )
    elif dataset == 'MNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)])

        train_data = datasets.MNIST(data_root, train=True, transform=transform_train, download=True)
        test_data = datasets.MNIST(data_root, train=False, transform=transform_test, download=True)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
        )
    elif dataset == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)])

        train_data = datasets.FashionMNIST(data_root, train=True, transform=transform_train, download=True)
        test_data = datasets.FashionMNIST(data_root, train=False, transform=transform_test, download=True)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=batch_size,
        )

    return train_data, test_data, train_loader, test_loader


def load_dvs10_data(batch_size, step, **kwargs):
    size = kwargs['size'] if 'size' in kwargs else 48
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
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=f'./dataset/dvs_cifar10/train_cache_{step}',
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(train_dataset,
                                     cache_path=f'./dataset/dvs_cifar10/test_cache_{step}',
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
    train_dataset = CustomDataset(train_dataset, np.array(indices_train))
    test_dataset = CustomDataset(test_dataset, np.array(indices_test))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, drop_last=False, num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=1
    )

    return train_loader, test_loader, train_dataset, test_dataset


def load_nmnist_data(batch_size, step, **kwargs):
    size = kwargs['size'] if 'size' in kwargs else 28
    sensor_size = tonic.datasets.NMNIST.sensor_size
    train_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        # tonic.transforms.DropEvent(p=0.1),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    test_transform = transforms.Compose([
        # tonic.transforms.Denoise(filter_time=10000),
        tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step), ])
    train_dataset = tonic.datasets.NMNIST(os.path.join(DATA_DIR, 'DVS/NMNIST'), transform=train_transform, train=True)
    test_dataset = tonic.datasets.NMNIST(os.path.join(DATA_DIR, 'DVS/NMNIST'), transform=test_transform, train=False)

    train_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),

    ])
    test_transform = transforms.Compose([
        lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
    ])

    train_dataset = DiskCachedDataset(train_dataset,
                                      cache_path=f'./dataset/NMNIST/train_cache_{step}',
                                      transform=train_transform)
    test_dataset = DiskCachedDataset(test_dataset,
                                     cache_path=f'./dataset/NMNIST/test_cache_{step}',
                                     transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=True, drop_last=False, num_workers=1
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=1
    )

    return train_loader, test_loader, train_dataset, test_dataset
