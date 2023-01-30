from .datasets import build_transform, build_dataset, get_mnist_data, get_fashion_data, \
    get_cifar10_data, get_cifar100_data, get_imnet_data, get_dvsg_data, get_dvsc10_data, \
    get_NCALTECH101_data, get_NCARS_data, get_nomni_data
from .utils import rescale, dvs_channel_check_expend

from .hmdb_dvs import HMDBDVS
from .ucf101_dvs import ucf101_dvs
from .ncaltech101 import NCALTECH101

__all__ = [
    'build_transform', 'build_dataset',
    'get_mnist_data', 'get_fashion_data', 'get_cifar10_data', 'get_cifar100_data', 'get_imnet_data',
    'get_dvsg_data', 'get_dvsc10_data', 'get_NCALTECH101_data', 'get_NCARS_data', 'get_nomni_data',
    'rescale', 'dvs_channel_check_expend'
]


dvs_data = [
    'dvsg',
    'dvsc10',
    'ncaltech101',
    'ncars',
    'dvsg',
    'ucf101dvs',
    'hmdbdvs',
]


def is_dvs_data(dataset):
    if dataset.lower() in dvs_data:
        return True
    else:
        return False
