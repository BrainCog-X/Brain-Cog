'''
Deep Residual Learning for Image Recognition
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import os
import sys
from functools import partial
from timm.models import register_model
from timm.models.layers import trunc_normal_, DropPath
from braincog.model_zoo.base_module import *
from braincog.base.node.node import *

__all__ = [
    'ResNet',
    'resnet18',
    'resnet34_half',
    'resnet34',
    'resnet50_half',
    'resnet50',
    'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2',
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    '''3x3 convolution with padding'''
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    '''1x1 convolution'''
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    """
    ResNet的基础模块, 采用identity-connection的方式.
    :param inplanes: 输出通道数
    :param planes: 内部通道数量
    :param stride: stride
    :param downsample: 是否降采样
    :param groups: 分组卷积
    :param base_width: 基础通道数量
    :param dilation: 空洞卷积
    :param norm_layer: Norm的方式
    :param node: 神经元类型, 默认为 ``LIFNode``
    """

    expansion = 1
    __constants__ = ['downsample']
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 node=LIFNode):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.node1 = node()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.relu = nn.ReLU(inplace=False)
        self.node2 = node()
        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.node1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.node2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class Bottleneck(nn.Module):
    """
    ResNet的Botteneck模块, 采用identity-connection的方式.
    :param inplanes: 输出通道数
    :param planes: 内部通道数量
    :param stride: stride
    :param downsample: 是否降采样
    :param groups: 分组卷积
    :param base_width: 基础通道数量
    :param dilation: 空洞卷积
    :param norm_layer: Norm的方式
    :param node: 神经元类型, 默认为 ``LIFNode``
    """
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 node=torch.nn.Identity):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv1x1(inplanes, width)

        self.bn2 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.bn3 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)

        # self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.node1 = node()
        self.node2 = node()
        self.node3 = node()

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.node1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.node2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.node3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class ResNet(BaseModule):
    """
    ResNet-SNN
    :param block: Block类型
    :param layers: block 层数
    :param inplanes: 输入通道数量
    :param num_classes: 输出类别数
    :param zero_init_residual: 是否使用零初始化
    :param groups: 卷积分组
    :param width_per_group: 每一组的宽度
    :param replace_stride_with_dilation: 是否使用stride替换dilation
    :param norm_layer: Norm 方式, 默认为 ``BatchNorm``
    :param step: 仿真步长, 默认为 ``8``
    :param encode_type: 编码方式, 默认为 ``direct``
    :param spike_output: 是否使用脉冲输出, 默认为 ``False``
    :param args:
    :param kwargs:
    """
    def __init__(self,
                 block,
                 layers,
                 inplanes=64,
                 num_classes=10,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 step=8,
                 encode_type='direct',
                 spike_output=False,
                 *args,
                 **kwargs):
        super().__init__(
            step,
            encode_type,
            *args,
            **kwargs
        )
        self.spike_output = spike_output
        self.num_classes = num_classes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # print('inplanes %d' % inplanes)
        self.inplanes = inplanes
        self.interplanes = [
            self.inplanes, self.inplanes * 2, self.inplanes * 4,
            self.inplanes * 8
        ]
        self.dilation = 1

        self.node = kwargs['node_type']
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs)

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.static_data = False

        self.dataset = kwargs['dataset']
        if self.dataset == 'dvsg' or self.dataset == 'dvsc10' or self.dataset == 'NCALTECH101' or self.dataset == 'NCARS' or self.dataset == 'DVSG':
            self.conv1 = nn.Conv2d(2 * self.init_channel_mul,
                                   self.inplanes,
                                   kernel_size=3,
                                   padding=1,
                                   bias=False)
        elif self.dataset == 'imnet':
            self.conv1 = nn.Conv2d(3 * self.init_channel_mul,
                                   self.inplanes,
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   bias=False)
            self.static_data = True
        elif self.dataset == 'cifar10' or self.dataset == 'cifar100':
            self.conv1 = nn.Conv2d(3 * self.init_channel_mul,
                                   self.inplanes,
                                   kernel_size=3,
                                   padding=1,
                                   bias=False)
            self.static_data = True

        # self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(
            block, self.interplanes[0], layers[0], node=self.node)
        self.layer2 = self._make_layer(block,
                                       self.interplanes[1],
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0], node=self.node)
        self.layer3 = self._make_layer(block,
                                       self.interplanes[2],
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1], node=self.node)
        self.layer4 = self._make_layer(block,
                                       self.interplanes[3],
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2], node=self.node)

        self.bn1 = norm_layer(self.inplanes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.spike_output:
            self.fc = nn.Linear(
                self.interplanes[3] * block.expansion, num_classes * 10)
            self.node2 = self.node()
            self.vote = VotingLayer(10)
        else:
            self.fc = nn.Linear(
                self.interplanes[3] * block.expansion, num_classes
            )
            self.node2 = nn.Identity()
            self.vote = nn.Identity()

        self.warm_up = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, node=torch.nn.Identity):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block == BasicBlock:
                downsample = nn.Sequential(
                    norm_layer(self.inplanes),
                    self.node(),
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                )
            elif block == Bottleneck:
                downsample = nn.Sequential(
                    norm_layer(self.inplanes),
                    self.node(),
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                )
            else:
                raise NotImplementedError

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer, node=node)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer, node=node))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:

            x = self.conv1(inputs)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.bn1(x)
            # x = self.node1(x)
            x = self.avgpool(x)

            x = torch.flatten(x, 1)
            # print(x.shape)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            x = self.node2(x)
            x = self.vote(x)

            return x

        else:
            outputs = []

            if self.warm_up:
                step = 1
            else:
                step = self.step
            for t in range(step):
                x = inputs[t]

                x = self.conv1(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.bn1(x)
                # x = self.node1(x)
                x = self.avgpool(x)

                x = torch.flatten(x, 1)
                x = self.fc(x)

                x = self.node2(x)
                x = self.vote(x)

                outputs.append(x)

            return sum(outputs) / len(outputs)


def _resnet(arch, block, layers, pretrained=False, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # only load state_dict()
    if pretrained:
        raise NotImplementedError

    return model


@register_model
def resnet9(pretrained=False, **kwargs):
    return _resnet('resnet9', BasicBlock, [1, 1, 1, 1], pretrained, **kwargs)


@register_model
def resnet18(pretrained=False, **kwargs):
    # kwargs['inplanes'] = 96
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


@register_model
def resnet34_half(pretrained=False, **kwargs):
    kwargs['inplanes'] = 32
    return _resnet('resnet34_half', BasicBlock, [3, 4, 6, 3], pretrained,
                   **kwargs)


@register_model
def resnet34(pretrained=False, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, **kwargs)


@register_model
def resnet50_half(pretrained=False, **kwargs):
    kwargs['inplanes'] = 32
    return _resnet('resnet50_half', Bottleneck, [3, 4, 6, 3], pretrained,
                   **kwargs)


@register_model
def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


@register_model
def resnet101(pretrained=False, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
                   **kwargs)


@register_model
def resnet152(pretrained=False, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
                   **kwargs)


@register_model
def resnext50_32x4d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained,
                   **kwargs)


@register_model
def resnext101_32x8d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained,
                   **kwargs)


@register_model
def wide_resnet50_2(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained,
                   **kwargs)


@register_model
def wide_resnet101_2(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained,
                   **kwargs)


if __name__ == '__main__':
    net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
    image_h, image_w = 224, 224
    from thop import profile
    from thop import clever_format

    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w),),
                            verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, flops: {flops}, params: {params},out_shape: {out.shape}')
