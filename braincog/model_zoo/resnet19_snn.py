# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/7/26 19:33
# User      : Floyed
# Product   : PyCharm
# Project   : braincog
# File      : resnet19_snn.py
# explain   :

import os
import sys
from functools import partial
import numpy as np
from timm.models import register_model
from timm.models.layers import trunc_normal_, DropPath
from braincog.model_zoo.base_module import *
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.datasets import is_dvs_data

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 node=LIFNode, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = ThresholdDependentBatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(num_features=planes, alpha=1.)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(num_features=planes, alpha=np.sqrt(.5))
        self.downsample = downsample
        self.stride = stride
        self.node1 = node()
        self.node2 = node()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.node1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.node2(out)

        return out



class ResNet(BaseModule):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1, width_per_group=128,
                 replace_stride_with_dilation=None, norm_layer=None, step=4, encode_type='direct', node_type=LIFNode,
                 *args, **kwargs):

        super().__init__(
            step,
            encode_type,
            *args,
            **kwargs
        )

        super().__init__(step, encode_type, *args, **kwargs)
        if not self.layer_by_layer:
            raise ValueError('ResNet-SNN only support for layer-wise mode, because of tdBN')

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if is_dvs_data(self.dataset):
            data_channel = 2
        else:
            data_channel = 3

        if norm_layer is None:
            norm_layer = ThresholdDependentBatchNorm2d
        self._norm_layer = partial(norm_layer,   step=step)
        self.sum_output=kwargs["sum_output"] if "sum_output"in kwargs else True 
        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(data_channel, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = self._norm_layer(num_features=self.inplanes, alpha=np.sqrt(.5))
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.node1 = self.node()
        self.node2 = self.node()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        # downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(num_features=planes * block.expansion, alpha=np.sqrt(.5)),
            )
        else:
            downsample = nn.Sequential(
                norm_layer(num_features=planes * block.expansion, alpha=np.sqrt(.5)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups,
                            base_width=self.base_width, norm_layer=norm_layer, node=self.node))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, node=self.node))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.node1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.node2(x)
        x = self.fc2(x)
        
        if self.sum_output:x= rearrange(x, '(t b) c -> b c t', t=self.step).mean(-1)
        else :x=  rearrange(x, '(t b) c -> t b c ', t=self.step)
        return x

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        return self._forward_impl(inputs)


def _resnet(arch, block, layers, pretrained, progress, norm=ThresholdDependentBatchNorm2d, **kwargs):
    tdBN = partial(norm, layer_by_layer=kwargs['layer_by_layer'], threshold=kwargs['threshold'])
    model = ResNet(block, layers, norm_layer=tdBN, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def resnet19(pretrained=False, progress=True, norm=ThresholdDependentBatchNorm2d, **kwargs):
    return _resnet('resnet19', BasicBlock, [3, 3, 2], pretrained, progress, norm=norm, **kwargs)


if __name__ == '__main__':
    net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format

    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w),),
                            verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, flops: {flops}, params: {params},out_shape: {out.shape}')
