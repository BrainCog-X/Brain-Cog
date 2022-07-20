import torch
from einops import repeat
from braincog.datasets.gen_input_signal import lambda_max


def rescale(x, factor=None):
    """
    数据放缩函数
    :param x: 输入的tensor
    :param factor: 缩放因子
    :return: 缩放后的数据
    """
    if factor:
        x *= factor
    else:
        x *= lambda_max
    return x


def dvs_channel_check_expend(x):
    """
    检查是否存在DVS数据缺失, N-Car中有的数据会缺少一个通道
    :param x: 输入的tensor
    :return: 补全之后的数据
    """
    if x.shape[1] == 1:
        return repeat(x, 'b c w h -> b (r c) w h', r=2)
    else:
        return x
