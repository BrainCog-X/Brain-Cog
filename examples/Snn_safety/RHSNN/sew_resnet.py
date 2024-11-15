import torch
import torch.nn as nn
from copy import deepcopy
import random

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
from braincog.base.node import *
from braincog.model_zoo.base_module import *
from braincog.datasets import is_dvs_data
from timm.models import register_model


def sew_function(x: torch.Tensor, y: torch.Tensor, cnf: str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, node: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.node1 = node()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.node2 = node()
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = node()
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.node1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.node2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(identity, out, self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, node: callable = None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.node1 = node()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.node2 = node()
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.node3 = node()
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = node()
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.node1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.node2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.node3(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(out, identity, self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


class SEWResNet19(BaseModule):
    def __init__(self, block, layers, num_classes=1000, step=8, encode_type="direct", zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, data_norm=None,
                 norm_layer=None, cnf: str = None, *args, **kwargs):
        super().__init__(
            step,
            encode_type,
            *args,
            **kwargs
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.normalize = data_norm
        self.node = kwargs['node_type']
        if issubclass(self.node, BaseNode):
            # self.node = partial(self.node, **kwargs, step=step)
            self.node1 = partial(self.node, **kwargs, step=step)()
            self.node2 = partial(self.node, **kwargs, step=step)
            self.node3 = partial(self.node, **kwargs, step=step)
            self.node4 = partial(self.node, **kwargs, step=step)
        self.once = kwargs["once"] if "once" in kwargs else False
        self.sum_output = kwargs["sum_output"] if "sum_output" in kwargs else True

        init_channel = 3

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

        self.conv1 = nn.Conv2d(init_channel, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        # self.node1 = self.node()
        self.layer1 = self._make_layer(block, 128, layers[0], cnf=cnf, node=self.node2, **kwargs)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, node=self.node3, **kwargs)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, node=self.node4, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str = None, node: callable = None,
                    **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, node, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, cnf=cnf, node=node, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, inputs):
        # See note [TorchScript super()]
        if self.normalize is not None:
            self.normalize.mean = self.normalize.mean.to(inputs.device)
            self.normalize.std = self.normalize.std.to(inputs.device)
            inputs = self.normalize(inputs)
        self.reset()

        if self.layer_by_layer:
            inputs = repeat(inputs, 'b c w h -> t b c w h', t=self.step)
            inputs = rearrange(inputs, 't b c w h -> (t b) c w h')
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.node1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)

            x = torch.flatten(x, 1)

            x = self.fc1(x)
            # x = self.node2(x)
            # x = self.fc2(x)

            x = rearrange(x, '(t b) c -> t b c', t=self.step)
            # print(x)
            if self.sum_output: x = x.mean(0)

            return x

    def _forward_once(self, x):
        # inputs = self.encoder(inputs)
        # x = inputs[t]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.node1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return x

    def forward(self, x):
        if self.once: return self._forward_once(x)
        return self._forward_impl(x)
