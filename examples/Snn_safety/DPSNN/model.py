import abc
from functools import partial
import torch
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule


class TEP(nn.Module):
    def __init__(self, step, channel, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TEP, self).__init__()
        self.step = step
        self.gn = nn.GroupNorm(channel, channel)


    def forward(self, x):

        x = rearrange(x, '(t b) c w h -> t b c w h', t=self.step)
        fire_rate = torch.mean(x, dim=0)
        fire_rate = self.gn(fire_rate) + 1

        x = x * fire_rate
        x = rearrange(x, 't b c w h -> (t b) c w h')

        return x


class BaseConvNet(BaseModule, abc.ABC):
    def __init__(self,
                 step,
                 input_channels,
                 num_classes,
                 encode_type,
                 spike_output: bool,
                 out_channels: list,
                 block_depth: list,
                 node_list: list,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.num_cls = num_classes
        self.spike_output = spike_output
        self.groups = kwargs['n_groups'] if 'n_groups' in kwargs else 1
        if not spike_output:
            node_list.append(nn.Identity)
            out_channels.append(self.num_cls)
            self.vote = nn.Identity()
            # self.vote = nn.Sequential(
            #     nn.Linear(self.step, 32),
            #     nn.ReLU(),
            #     nn.Linear(32, 1)
            # )
        else:
            out_channels.append(10 * self.num_cls)
            self.vote = VotingLayer(10)

        # check list length
        if len(node_list) != len(out_channels):
            raise ValueError
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.block_depth = block_depth
        self.node_list = node_list
        self.feature = self._create_feature()
        self.fc = self._create_fc()
        if self.layer_by_layer:
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = nn.Flatten()

    @staticmethod
    def _create_feature(self):
        raise NotImplementedError

    @staticmethod
    def _create_fc(self):
        raise NotImplementedError

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()
        if not self.training:
            self.fire_rate.clear()

        if not self.layer_by_layer:
            outputs = []
            if self.warm_up:
                step = 1
            else:
                step = self.step

            for t in range(step):
                x = inputs[t]
                x = self.feature(x)
                x = self.flatten(x)
                x = self.fc(x)
                x = self.vote(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)
            # outputs = torch.stack(outputs)
            # outputs = rearrange(outputs, 't b c -> b c t')
            # outputs = self.vote(outputs).squeeze()
            # return outputs

        else:
            x = self.feature(inputs)
            x = self.flatten(x)
            x = self.fc(x)
            if self.groups == 1:
                x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            else:
                x = rearrange(x, 'b (c t) -> t b c', t=self.step).mean(0)
            x = self.vote(x)
            return x


class LayerWiseConvModule(nn.Module):
    """
    SNN卷积模块
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param kernel_size: kernel size
    :param stride: stride
    :param padding: padding
    :param bias: Bias
    :param node: 神经元类型
    :param kwargs:
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 bias=False,
                 node=LIFNode,
                 step=6,
                 **kwargs):

        super().__init__()

        if node is None:
            raise TypeError

        self.groups = kwargs['groups'] if 'groups' in kwargs else 1
        self.conv = nn.Conv2d(in_channels=in_channels * self.groups,
                              out_channels=out_channels * self.groups,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride,
                              bias=bias)
        self.gn = nn.GroupNorm(16, out_channels * self.groups)
        self.node = partial(node, **kwargs)()
        self.step = step
        self.activation = nn.Identity()

    def forward(self, x):
        x = rearrange(x, '(t b) c w h -> t b c w h', t=self.step)
        outputs = []

        for t in range(self.step):
            outputs.append(self.gn(self.conv(x[t])))
        outputs = torch.stack(outputs)  # t b c w h
        outputs = rearrange(outputs, 't b c w h -> (t b) c w h')
        outputs = self.node(outputs)

        return outputs


class LayerWiseLinearModule(nn.Module):
    """
    线性模块
    :param in_features: 输入尺寸
    :param out_features: 输出尺寸
    :param bias: 是否有Bias, 默认 ``False``
    :param node: 神经元类型, 默认 ``LIFNode``
    :param args:
    :param kwargs:
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 node=LIFNode,
                 step=6,
                 spike=False,
                 *args,
                 **kwargs):
        super().__init__()
        if node is None:
            raise TypeError

        self.groups = kwargs['groups'] if 'groups' in kwargs else 1
        if self.groups == 1:
            self.fc = nn.Linear(in_features=in_features,
                                out_features=out_features, bias=bias)
        else:
            self.fc = nn.ModuleList()
            for i in range(self.groups):
                self.fc.append(nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias
                ))
        self.node = partial(node, **kwargs)()
        self.step = step
        self.spike = spike

    def forward(self, x):
        if self.groups == 1:  # (t b) c
            x = rearrange(x, '(t b) c -> t b c', t=self.step)
            outputs = []
            for t in range(self.step):
                outputs.append(self.fc(x[t]))
            outputs = torch.stack(outputs)  # t b c
            outputs = rearrange(outputs, 't b c -> (t b) c')

        else:  # b (c t)
            x = rearrange(x, 'b (c t) -> t b c', t=self.groups)
            outputs = []
            for i in range(self.groups):
                outputs.append(self.fc[i](x[i]))
            outputs = torch.stack(outputs)  # t b c
            outputs = rearrange(outputs, 't b c -> b (c t)')
        if self.spike:
            return self.node(outputs)
        else:
            return outputs


class LayWiseConvNet(BaseConvNet):
    def __init__(self,
                 step,
                 input_channels,
                 num_classes,
                 encode_type,
                 spike_output: bool,
                 out_channels: list,
                 node_list: list,
                 block_depth: list,
                 *args,
                 **kwargs):
        super().__init__(step,
                         input_channels,
                         num_classes,
                         encode_type,
                         spike_output,
                         out_channels,
                         block_depth,
                         node_list,
                         *args,
                         **kwargs)

    def _create_feature(self):
        feature_depth = len(self.node_list) - 1

        feature = [LayerWiseConvModule(
            self.input_channels * self.init_channel_mul, self.out_channels[0], node=self.node_list[0],
            groups=self.groups, step=self.step)]
        if self.block_depth[0] != 1:
            feature.extend(
                [LayerWiseConvModule(self.out_channels[0], self.out_channels[0], node=self.node_list[0],
                                     groups=self.groups, step=self.step)] * (
                        self.block_depth[0] - 1),
            )
        feature.append(TEP(channel=self.out_channels[0], step=self.step))
        feature.append(nn.AvgPool2d(kernel_size=4, stride=2))
        for i in range(1, feature_depth - 1):
            feature.append(LayerWiseConvModule(
                self.out_channels[i - 1], self.out_channels[i], node=self.node_list[i], groups=self.groups,
                step=self.step))
            if self.block_depth[i] != 1:
                feature.extend(
                    [LayerWiseConvModule(self.out_channels[i], self.out_channels[i], node=self.node_list[i],
                                         groups=self.groups,
                                         step=self.step)] * (
                            self.block_depth[i] - 1),
                )
            feature.append(TEP(channel=self.out_channels[i], step=self.step))
            feature.append(nn.AvgPool2d(kernel_size=4, stride=2))
        feature.append(LayerWiseConvModule(
            self.out_channels[-3], self.out_channels[-2], node=self.node_list[-2], groups=self.groups,
            step=self.step))
        if self.block_depth[feature_depth - 1] != 1:
            feature.extend(
                [LayerWiseConvModule(self.out_channels[-2], self.out_channels[-2], node=self.node_list[-2],
                                     groups=self.groups,
                                     step=self.step)] * (
                        self.block_depth[feature_depth - 1] - 1),
            )
        feature.append(nn.AdaptiveAvgPool2d(1))

        return nn.Sequential(*feature)

    def _create_fc(self):
        fc = nn.Sequential(
            # NDropout(.5),
            LayerWiseLinearModule(
                self.out_channels[-2], self.out_channels[-1], node=self.node_list[-1], groups=self.groups,
                step=self.step, spike=False)
        )
        return fc


@register_model
def cifar_convnet(step,
                encode_type,
                spike_output: bool,
                node_type,
                *args,
                **kwargs):
    # out_channels = [256, 256, 512, 1024]
    out_channels = [64, 128, 128, 256]
    block_depth = [2, 2, 2, 2]
    # print(kwargs)
    node_cls = partial(node_type, step=step, **kwargs)
    # print(node_cls)
    if spike_output:
        node_list = [node_cls] * (len(out_channels) + 1)
    else:
        node_list = [node_cls] * (len(out_channels))

    return LayWiseConvNet(step=step,
                          input_channels=3,
                          encode_type=encode_type,
                          node_list=node_list,
                          block_depth=block_depth,
                          out_channels=out_channels,
                          spike_output=spike_output,
                          **kwargs)


@register_model
def dvs_convnet(step,
                encode_type,
                spike_output: bool,
                node_type,
                num_classes,
                *args,
                **kwargs):
    out_channels = [64, 128, 256, 512, 1024]
    block_depth = [2, 1, 2, 1, 2]

    node_cls = partial(node_type, step=step, **kwargs)
    if spike_output:
        node_list = [node_cls] * (len(out_channels) + 1)
        # node_list[-2] = partial(DoubleSidePLIFNode, step=step, **kwargs)
    else:
        node_list = [node_cls] * (len(out_channels))
        # node_list[-1] = partial(DoubleSidePLIFNode, step=step, **kwargs)

    return LayWiseConvNet(step=step,
                          input_channels=2,
                          num_classes=num_classes,
                          encode_type=encode_type,
                          node_list=node_list,
                          block_depth=block_depth,
                          out_channels=out_channels,
                          spike_output=spike_output,
                          **kwargs)


@register_model
class SimpleSNN(BaseModule, abc.ABC):
    def __init__(self,
                 channel=1,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.num_classes = num_classes

        self.node = node_type
        init_channel = channel

        self.feature = nn.Sequential(
            LayerWiseConvModule(init_channel, 32, kernel_size=7, padding=0, node=self.node, step=step),
            TEP(step=step, channel=32),
            nn.AvgPool2d(kernel_size=2, stride=2),
            LayerWiseConvModule(32, 64, kernel_size=4, padding=0, node=self.node, step=step),
            TEP(step=step, channel=64),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            LayerWiseLinearModule(64 * 4 * 4, self.num_classes, node=self.node, spike=False, step=step),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)
