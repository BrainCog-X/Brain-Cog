import torch
import torch.nn as nn


def mergeConvBN(m):
    """
    合并网络模块中的卷积与BN层
    """
    children = list(m.named_children())
    c, cn = None, None

    for i, (name, child) in enumerate(children):
        if isinstance(child, nn.BatchNorm2d):
            bc = merge(c, child)
            m._modules[cn] = bc
            m._modules[name] = torch.nn.Identity()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            mergeConvBN(child)
    return m


def merge(conv, bn):
    """
    conv: 卷积层实例
    bn: BN层实例
    """
    w = conv.weight
    mean, var_sqrt, beta, gamma = bn.running_mean, torch.sqrt(bn.running_var + bn.eps), bn.weight, bn.bias
    b = conv.bias if conv.bias is not None else mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv
