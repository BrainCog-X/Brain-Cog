import numpy as np
from timm.models import register_model
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.base.node.node import *
from braincog.base.connection.layer import *


class MNISTNet(BaseModule):
    def __init__(self, step=20, encode_type='rate', if_back=True, if_ei=True, data='mnist', *args, **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.if_back = if_back
        self.if_ei = if_ei
        if data == 'mnist':
            self.cfg_conv = ((1, 15, 5, 1, 0), (15, 40, 5, 1, 0))
            self.cfg_fc = (300, 10)
            self.cfg_kernel = (24, 8, 4)
            cfg_backei = 2
        if data == 'fashion':
            self.cfg_conv = ((1, 32, 5, 1, 2), (32, 64, 5, 1, 2))
            self.cfg_fc = (1024, 10)
            self.cfg_kernel = (28, 14, 7)
            cfg_backei = 1
        self.feature = nn.Sequential(
            nn.Conv2d(self.cfg_conv[0][0], self.cfg_conv[0][1], self.cfg_conv[0][2], self.cfg_conv[0][3],
                      self.cfg_conv[0][4]),
            BackEINode(channel=self.cfg_conv[0][1], if_back=self.if_back, if_ei=self.if_ei, cfg_backei=cfg_backei),
            nn.AvgPool2d(2),
            nn.Conv2d(self.cfg_conv[1][0], self.cfg_conv[1][1], self.cfg_conv[1][2], self.cfg_conv[1][3],
                      self.cfg_conv[1][4]),
            BackEINode(channel=self.cfg_conv[1][1], if_back=self.if_back, if_ei=self.if_ei, cfg_backei=cfg_backei),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(self.cfg_kernel[2] * self.cfg_kernel[2] * self.cfg_conv[1][1], self.cfg_fc[0]),
            BackEINode(if_back=False, if_ei=False),
            nn.Linear(self.cfg_fc[0], self.cfg_fc[1]),
            BackEINode(if_back=False, if_ei=False)
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()
        if not self.training:
            self.fire_rate.clear()
        outputs = []
        step = self.step
        for t in range(step):
            x = inputs[t]
            x = self.feature(x)
            outputs.append(x)

        return sum(outputs) / len(outputs)


class CIFARNet(BaseModule):
    def __init__(self, step=20, encode_type='rate', if_back=True, if_ei=True, *args, **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.if_back = if_back
        self.if_ei = if_ei
        self.feature = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            BackEINode(channel=128, if_back=self.if_back, if_ei=self.if_ei, cfg_backei=1),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            BackEINode(channel=256, if_back=self.if_back, if_ei=self.if_ei, cfg_backei=1),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            BackEINode(channel=512, if_back=self.if_back, if_ei=self.if_ei, cfg_backei=1),
            nn.Dropout(0.5),
            nn.AvgPool2d(2),

            nn.Flatten(),
            nn.Linear(4 * 4 * 512, 1024),
            BackEINode(if_back=False, if_ei=False),
            nn.Dropout(0.5),

            nn.Linear(1024, 10),
            BackEINode(if_back=False, if_ei=False)
        )

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()
        if not self.training:
            self.fire_rate.clear()
        outputs = []
        step = self.step
        for t in range(step):
            x = inputs[t]
            x = self.feature(x)
            outputs.append(x)

        return sum(outputs) / len(outputs)
