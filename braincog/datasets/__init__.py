from .datasets import build_transform, build_dataset, get_mnist_data, get_fashion_data, \
    get_cifar10_data, get_cifar100_data, get_imnet_data, get_dvsg_data, get_dvsc10_data, \
    get_NCALTECH101_data, get_NCARS_data, get_nomni_data
from .utils import rescale, dvs_channel_check_expend

__all__ = [
    'build_transform', 'build_dataset',
    'get_mnist_data', 'get_fashion_data', 'get_cifar10_data', 'get_cifar100_data', 'get_imnet_data',
    'get_dvsg_data', 'get_dvsc10_data', 'get_NCALTECH101_data', 'get_NCARS_data', 'get_nomni_data',
    'rescale', 'dvs_channel_check_expend'
]


def is_dvs_data(dataset):
    if dataset == 'dvsg' or dataset == 'dvsc10' or dataset == 'NCALTECH101' or dataset == 'NCARS' or dataset == 'DVSG':
        return True
    else:
        return False
