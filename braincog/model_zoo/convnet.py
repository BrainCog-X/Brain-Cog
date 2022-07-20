import abc
from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule


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


class MNISTConvNet(BaseConvNet):
    def __init__(self,
                 step,
                 input_channels,
                 num_classes,
                 encode_type,
                 block_depth,
                 spike_output: bool,
                 out_channels: list,
                 node_list: list,
                 *args,
                 **kwargs):
        self.feature_size = 28
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
        feature_depth = len(self.node_list) - 2

        feature = [BaseConvModule(
            self.input_channels, self.out_channels[0], node=self.node_list[0])]
        if self.block_depth[0] != 1:
            feature.extend(
                [BaseConvModule(self.out_channels[0], self.out_channels[0], node=self.node_list[0])] * (
                    self.block_depth[0] - 1),
            )
        feature.append(nn.AvgPool2d(2))
        self.feature_size = self.feature_size // 2

        for i in range(1, feature_depth):
            feature.append(BaseConvModule(
                self.out_channels[i - 1], self.out_channels[i], node=self.node_list[i]))
            if self.block_depth[i] != 1:
                feature.extend(
                    [BaseConvModule(self.out_channels[i], self.out_channels[i], node=self.node_list[i])] * (
                        self.block_depth[0] - 1),
                )
            feature.append(nn.AvgPool2d(2))
            feature.append(self.node_list[0]())
            self.feature_size = self.feature_size // 2

        return nn.Sequential(*feature)

    def _create_fc(self):
        fc = nn.Sequential(
            NDropout(.5),
            BaseLinearModule(self.out_channels[-3] * self.feature_size * self.feature_size, self.out_channels[-2],
                             node=self.node_list[-2]),
            NDropout(.5),
            BaseLinearModule(
                self.out_channels[-2], self.out_channels[-1], node=self.node_list[-1])
        )
        return fc


class CifarConvNet(BaseConvNet):
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

        feature = [BaseConvModule(
            self.input_channels * self.init_channel_mul, self.out_channels[0], node=self.node_list[0], groups=self.groups)]
        if self.block_depth[0] != 1:
            feature.extend(
                [BaseConvModule(self.out_channels[0], self.out_channels[0], node=self.node_list[0], groups=self.groups)] * (
                    self.block_depth[0] - 1),
            )
        feature.append(nn.AvgPool2d(2))
        for i in range(1, feature_depth - 1):
            feature.append(BaseConvModule(
                self.out_channels[i - 1], self.out_channels[i], node=self.node_list[i], groups=self.groups))
            if self.block_depth[i] != 1:
                feature.extend(
                    [BaseConvModule(self.out_channels[i], self.out_channels[i], node=self.node_list[i], groups=self.groups)] * (
                        self.block_depth[i] - 1),
                )
            feature.append(nn.AvgPool2d(2))

        feature.append(BaseConvModule(
            self.out_channels[-3], self.out_channels[-2], node=self.node_list[-2], groups=self.groups))
        if self.block_depth[feature_depth - 1] != 1:
            feature.extend(
                [BaseConvModule(self.out_channels[-2], self.out_channels[-2], node=self.node_list[-2], groups=self.groups)] * (
                    self.block_depth[feature_depth - 1] - 1),
            )
        feature.append(nn.AdaptiveAvgPool2d((1, 1)))

        return nn.Sequential(*feature)

    def _create_fc(self):
        fc = nn.Sequential(
            # NDropout(.5),
            BaseLinearModule(
                self.out_channels[-2], self.out_channels[-1], node=self.node_list[-1], groups=self.groups)
        )
        return fc


@register_model
def mnist_convnet(step,
                  encode_type,
                  spike_output: bool,
                  node_type,
                  *args,
                  **kwargs):
    out_channels = [128, 128, 2048]
    block_depth = [1, 1]
    node_cls = partial(node_type, step=step, **kwargs)
    if spike_output:
        node_list = [node_cls] * (len(out_channels) + 1)
    else:
        node_list = [node_cls] * (len(out_channels))

    return MNISTConvNet(step=step,
                        input_channels=1,
                        encode_type=encode_type,
                        block_depth=block_depth,
                        node_list=node_list,
                        out_channels=out_channels,
                        spike_output=spike_output,
                        **kwargs)


@register_model
def cifar_convnet(step,
                  encode_type,
                  spike_output: bool,
                  node_type,
                  *args,
                  **kwargs):
    out_channels = [256, 256, 512, 1024]
    # out_channels = [64, 128, 128, 256]
    block_depth = [2, 2, 2, 2]
    # print(kwargs)
    node_cls = partial(node_type, step=step, **kwargs)
    # print(node_cls)
    if spike_output:
        node_list = [node_cls] * (len(out_channels) + 1)
    else:
        node_list = [node_cls] * (len(out_channels))

    return CifarConvNet(step=step,
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
    out_channels = [128, 256, 256, 512, 512]
    block_depth = [2, 1, 2, 1, 2]

    # out_channels = [40, 80, 80, 160, 160]
    # out_channels = [256, 512, 512, 1024, 1024]
    # out_channels = [64, 128, 128, 256, 256]
    # block_depth = [4, 2, 4, 2, 4]

    # out_channels = [128, 256, 512, 512]
    # block_depth = [2, 2, 2, 2]
    node_cls = partial(node_type, step=step, **kwargs)
    if spike_output:
        node_list = [node_cls] * (len(out_channels) + 1)
        # node_list[-2] = partial(DoubleSidePLIFNode, step=step, **kwargs)
    else:
        node_list = [node_cls] * (len(out_channels))
        # node_list[-1] = partial(DoubleSidePLIFNode, step=step, **kwargs)

    return CifarConvNet(step=step,
                        input_channels=2,
                        num_classes=num_classes,
                        encode_type=encode_type,
                        node_list=node_list,
                        block_depth=block_depth,
                        out_channels=out_channels,
                        spike_output=spike_output,
                        **kwargs)
