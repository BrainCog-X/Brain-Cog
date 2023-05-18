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

from braincog.datasets import is_dvs_data
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule

@register_model
class SNN(BaseModule):
    def __init__(self,
                 num_classes=100,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 batch_size=100,
                 task_num=10,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False

        self.num_classes = num_classes
        self.batch_size=batch_size
        self.num_classes = num_classes
        self.task_num=task_num
        self.out_num=int(self.num_classes/self.task_num)

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        if not is_dvs_data(self.dataset):
            init_channel = 3
            output_size = 2
        else:
            init_channel = 2
            output_size = 3
        self.channel_number=[128,256]
        self.fcnum=512
        #self.channel_number=[512,1024,2048]

        self.feature = nn.Sequential(
            BaseConvModule(1, self.channel_number[0], kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
            BaseConvModule(self.channel_number[0], self.channel_number[1], kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.AvgPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            BaseLinearModule(
                self.channel_number[1]*7*7, self.fcnum, node=self.node),
        )

        self.dec = nn.ModuleDict()
        for task in range(self.task_num):
            ta=str(task)
            self.dec[ta] = self._create_decision()


    def logits(self, x):
        outputs =torch.zeros((self.task_num, self.batch_size,self.out_num),device='cuda')
        for task, func in self.dec.items():
            ta=int(task)
            outputs[ta]=func(x)
        return outputs

    def _create_decision(self):
        fc = nn.Linear(self.fcnum, self.out_num)
        #fc = BaseLinearModule(self.fcnum, 10, node=self.node)
        return fc

    def forward(self, inputs, mat):
        inputs = self.encoder(inputs)
        self.reset()
        step = self.step
        outputs = []
        for index, item in enumerate(self.parameters()):
            if len(item.size()) > 1:
                ww=item.data
                item.data=ww*mat[index].cuda()

        for t in range(step):
            x = inputs[t]
            x = self.feature(x)
            x = self.fc(x)
            x = self.logits(x)
            outputs.append(x)

        out=sum(outputs).cuda()

        return out / step

class Taskmodel(BaseModule):
    def __init__(self,step=0,encode_type='direct',num_classes=10,task_num=10,
                 batch_size=100,*args,**kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.out_num=int(num_classes/task_num)

        self.flat=torch.zeros((batch_size,100),device='cuda')
        self.fctask1=nn.Linear(100,100)
        self.relu=nn.ReLU()
        self.fctask2=nn.Linear(100,self.out_num)

    def forward(self, out):

        for i in range(len(out)):
            self.flat[:,self.out_num*i:self.out_num*(i+1)]=out[i]

        x=self.fctask1(self.flat)
        x=self.relu(x)
        predict_task=self.fctask2(x)

        return predict_task


# class MaskConvModule(nn.Module):
#     """
#     SNN卷积模块
#     :param in_channels: 输入通道数
#     :param out_channels: 输出通道数
#     :param kernel_size: kernel size
#     :param stride: stride
#     :param padding: padding
#     :param bias: Bias
#     :param node: 神经元类型
#     :param kwargs:
#     """
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size=(3, 3),
#                  stride=(1, 1),
#                  padding=(1, 1),
#                  bias=False,
#                  node=PLIFNode,
#                  **kwargs):

#         super().__init__()

#         if node is None:
#             raise TypeError

#         self.groups = kwargs['groups'] if 'groups' in kwargs else 1
#         self.conv = MConv2d(in_channels=in_channels * self.groups,
#                               out_channels=out_channels * self.groups,
#                               kernel_size=kernel_size,
#                               padding=padding,
#                               stride=stride,
#                               bias=bias)

#         self.bn = nn.BatchNorm2d(out_channels * self.groups)

#         self.node = partial(node, **kwargs)()

#         self.activation = nn.Identity()

#     def forward(self, x, mat):
#         x = self.conv(x,mat)
#         x = self.bn(x)
#         x = self.node(x)
#         return x

# class MConv2d(nn.Conv2d):

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, gain=True):
#         super(MConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                        padding, dilation, groups, bias)

#         self.gain = 1.

#     def forward(self, x, mat):
#         weight = self.weight
#         weight = weight*mat
#         return F.conv2d(x, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)

# class MaskLinearModule(nn.Module):
#     """
#     线性模块
#     :param in_features: 输入尺寸
#     :param out_features: 输出尺寸
#     :param bias: 是否有Bias, 默认 ``False``
#     :param node: 神经元类型, 默认 ``LIFNode``
#     :param args:
#     :param kwargs:
#     """
#     def __init__(self,
#                  in_features: int,
#                  out_features: int,
#                  bias=True,
#                  node=LIFNode,
#                  *args,
#                  **kwargs):
#         super().__init__()
#         if node is None:
#             raise TypeError

#         self.fc = MLinear(in_features=in_features,
#                                 out_features=out_features, bias=bias)
#         self.node = partial(node, **kwargs)()

#     def forward(self, x,mat):
#         outputs = self.fc(x,mat)
#         return self.node(outputs)

# class MLinear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True):
#         super(MLinear, self).__init__(in_features, out_features, bias)
#         self.gain = 1.

#     def forward(self, input, mat):
#         weight = self.weight
#         weight = weight*mat
#         return F.linear(input, weight, self.bias)