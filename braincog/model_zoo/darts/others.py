# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/5/22 13:32
# User      : yu
# Product   : PyCharm
# Project   : BrainCog
# File      : others.py
# explain   :
from functools import partial
import torch
import torch.nn as nn
from copy import deepcopy

from timm.models import register_model

from braincog.base.node.node import *
from braincog.base.connection.layer import WSConv2d
from braincog.datasets import is_dvs_data
from braincog.model_zoo.base_module import BaseModule, BaseConvModule

@register_model
class CIFARNet_Wu(BaseModule):

    def __init__(
            self, num_classes=10,
            node_type=LIFNode,
            step=4,
            encode_type='direct',
            *args,
            **kwargs,
    ):
        super().__init__(step, encode_type, *args, **kwargs)
        self.dataset = kwargs['dataset']
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        channels = 32
        if not is_dvs_data(self.dataset):
            init_channel = 3
            out_size = 2 ** 2
        else:
            init_channel = 2
            out_size = 3 ** 2

        self.feature = nn.Sequential(
            BaseConvModule(init_channel, channels, node=self.node),
            BaseConvModule(channels, channels * 2, node=self.node),
            nn.AvgPool2d(2, 2),
            self.node(),
            BaseConvModule(channels * 2, channels * 4, node=self.node),
            nn.AvgPool2d(2, 2),
            # self.node(),
            BaseConvModule(channels * 4, channels * 8, node=self.node),
            BaseConvModule(channels * 8, channels * 4, node=self.node),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(channels * 4 * 8 * 8, channels * 8, bias=False),
            self.node(),
            nn.Linear(channels * 8, channels * 4, bias=False),
            self.node(),
            nn.Linear(channels * 4, num_classes, bias=False)
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs).contiguous()
        self.reset()
        outputs = []
        for t in range(self.step):
            x = inputs[t]
            x = self.feature(x)
            x = self.fc(x)
            outputs.append(x)
        return sum(outputs) / len(outputs)

@register_model
class CIFARNet_Fang(BaseModule):

    def __init__(
            self, num_classes=10,
            node_type=LIFNode,
            step=4,
            encode_type='direct',
            *args,
            **kwargs,
    ):
        super().__init__(step, encode_type, *args, **kwargs)
        self.dataset = kwargs['dataset']
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        channels = 32
        if not is_dvs_data(self.dataset):
            init_channel = 3
        else:
            init_channel = 2

        self.feature = nn.Sequential(
            BaseConvModule(init_channel, channels, node=self.node),
            BaseConvModule(channels, channels, node=self.node),
            BaseConvModule(channels, channels, node=self.node),
            nn.MaxPool2d(2, 2),
            BaseConvModule(channels, channels, node=self.node),
            BaseConvModule(channels, channels, node=self.node),
            BaseConvModule(channels, channels, node=self.node),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(channels * 8 * 8, channels * 8, bias=False),
            self.node(),
            nn.Linear(channels * 8, channels, bias=False),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs).contiguous()
        self.reset()

        outputs = []
        for t in range(self.step):
            x = inputs[t]
            x = self.feature(x)
            x = self.fc(x)
            outputs.append(x)
        return sum(outputs) / len(outputs)

@register_model
class DVS_CIFARNet_Fang(BaseModule):

    def __init__(
            self, num_classes=10,
            node_type=LIFNode,
            step=10,
            encode_type='direct',
            *args,
            **kwargs,
    ):
        super().__init__(step, encode_type, *args, **kwargs)
        self.dataset = kwargs['dataset']
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        channels = 128
        if not is_dvs_data(self.dataset):
            init_channel = 3
        else:
            init_channel = 2

        self.feature = nn.Sequential(
            BaseConvModule(init_channel, channels, node=self.node),
            nn.MaxPool2d(2, 2),
            BaseConvModule(channels, channels, node=self.node),
            nn.MaxPool2d(2, 2),
            BaseConvModule(channels, channels, node=self.node),
            nn.MaxPool2d(2, 2),
            BaseConvModule(channels, channels, node=self.node),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(channels * 8 * 8, channels * 4, bias=False),
            self.node(),
            nn.Linear(channels * 4, channels, bias=False),
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs).contiguous()
        self.reset()

        outputs = []
        for t in range(self.step):
            x = inputs[t]
            x = self.feature(x)
            x = self.fc(x)
            outputs.append(x)
        return sum(outputs) / len(outputs)