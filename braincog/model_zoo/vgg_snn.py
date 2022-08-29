# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/7/26 18:56
# User      : Floyed
# Product   : PyCharm
# Project   : BrainCog
# File      : vgg_snn.py
# explain   :

from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.datasets import is_dvs_data


@register_model
class SNN7_tiny(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        assert not is_dvs_data(self.dataset), 'SNN7_tiny only support static datasets now'

        self.feature = nn.Sequential(
            BaseConvModule(3, 16, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(16, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, self.num_classes),
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


@register_model
class SNN5(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if not is_dvs_data(self.dataset):
            init_channel = 3
        else:
            init_channel = 2

        self.feature = nn.Sequential(
            BaseConvModule(init_channel, 16, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
            BaseConvModule(16, 64, kernel_size=(5, 5), padding=(2, 2), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(64, 128, kernel_size=(5, 5), padding=(2, 2), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node, n_preact=self.n_preact),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, self.num_classes),
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


@register_model
class VGG_SNN(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if not is_dvs_data(self.dataset):
            raise NotImplementedError('VGG-SNN model is only for DVS data, but current datasets is {}'.format(self.dataset))

        self.feature = nn.Sequential(
            BaseConvModule(2, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, self.num_classes),
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
