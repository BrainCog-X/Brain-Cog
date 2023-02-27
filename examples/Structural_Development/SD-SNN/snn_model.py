import abc
from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model

from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class my_cifar_model(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.num_classes = num_classes

        self.feature = nn.Sequential(
            BaseConvModule(3, 128, kernel_size=(3, 3), padding=(1, 1)),
            BaseConvModule(128,128, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(2),
            BaseConvModule(128,256, kernel_size=(3, 3), padding=(1, 1)),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1)),
            BaseConvModule(512, 512, kernel_size=(3, 3), padding=(1, 1)),
        )

        self.convlayer = [0,1,3,4,6,7,8]

        self.cfla=self._cflatten()
        self.fc_prun = self._create_fc_prun()
        self.fc = self._create_fc()

    def _cflatten(self):
        fc = nn.Sequential(
            nn.Flatten(),
        )
        return fc
        
    def _create_fc_prun(self):
        fc = nn.Sequential(
            BaseLinearModule(512*8*8, 512)
        )
        return fc

    def _create_fc(self):
        fc = nn.Sequential(
            BaseLinearModule(512, self.num_classes)
        )
        return fc
    
    def forward(self, inputs):
        inputs = self.encoder(inputs)

        self.reset()
        if not self.training:
            self.fire_rate.clear()

        outputs = []
                
        for t in range(self.step):
            x = inputs[t]
            if x.shape[-1] > 32:
                x = F.interpolate(x, size=[64, 64])

            for i in range(len(self.feature)):
                x=self.feature[i](x)

            x=self.cfla(x)
            x=self.fc_prun(x)
            x = self.fc(x)

            outputs.append(x)

        return sum(outputs) / len(outputs)


