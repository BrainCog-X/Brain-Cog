import torch
import torch.nn as nn
from copy import deepcopy

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url
from braincog.base.node import *
from braincog.model_zoo.base_module import *
from braincog.datasets import is_dvs_data
from timm.models import register_model
__all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101',
           'sew_resnet152', 'sew_resnext50_32x4d', 'sew_resnext101_32x8d',
           'sew_wide_resnet50_2', 'sew_wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def sew_function(x: torch.Tensor, y: torch.Tensor, cnf:str):
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
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, planes_cur,stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None, cnf: str = None, node: callable = LIFNode, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv=nn.Sequential(
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        BaseConvModule(inplanes, planes_cur, kernel_size=(3, 3), stride=stride,padding=(1, 1), node=node),
        BaseConvModule(planes, planes_cur, kernel_size=(3, 3), padding=(1, 1), node=node),
        downsample,
        BaseConvModule(planes, planes_cur, kernel_size=(3, 3), padding=(1, 1), node=node),
        BaseConvModule(planes, planes_cur, kernel_size=(3, 3), padding=(1, 1), node=node),)
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(identity, out, self.cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'



class SEWResNet(BaseModule):
    def __init__(self, block, layers, c_dim=[64,128,256,512], cdim_cur=[],step=4,encode_type="direct",zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cnf: str =  'ADD',   *args,**kwargs):
        super().__init__(            
            step,
            encode_type,
            *args,
            **kwargs
        )
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups=groups

        self.node = LIFNode
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)
        self.c_dim=c_dim
        if len(cdim_cur)>0:
            self.cdim_cur=cdim_cur
        else:
            self.cdim_cur=self.c_dim
        self.inplanes = c_dim[0]
        self.inplanes_cur = cdim_cur[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

 

        self.conv1 = nn.Conv2d(3, self.cdim_cur[0], kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.cdim_cur[0])
        self.node1 = self.node()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_convnets = nn.ModuleList()
        self.layer_convnets.append(self._make_layer(block, c_dim[0], self.cdim_cur[0],layers[0], cnf=cnf, node=self.node, **kwargs))
        self.layer_convnets.append(self._make_layer(block, c_dim[1], self.cdim_cur[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, node=self.node, **kwargs))
        self.layer_convnets.append(self._make_layer(block, c_dim[2], self.cdim_cur[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, node=self.node, **kwargs))
        self.layer_convnets.append(self._make_layer(block, c_dim[3], self.cdim_cur[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, node=self.node, **kwargs))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.classifer=None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, planes_cur,blocks, stride=1, dilate=False, cnf: str=None, node: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_cur, planes_cur * block.expansion, stride),
                norm_layer(planes_cur * block.expansion),
                node()
            )

        layers =block(self.inplanes, planes, planes_cur,stride, downsample, self.groups,
                             previous_dilation, norm_layer, cnf, node, **kwargs)
        self.inplanes = planes * block.expansion
        self.inplanes_cur= planes_cur * block.expansion
        # for _ in range(1, blocks):
        #     layers.append(block(self.inplanes, planes, groups=self.groups,
        #                         dilation=self.dilation,
        #                         norm_layer=norm_layer, cnf=cnf, node=node, **kwargs))

        return layers

    def forward_init(self, inputs):
        # See note [TorchScript super()]
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.node1(x)
        # x = self.maxpool(x)
        return x

    
    def forward_impl(self, inputs):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x



def _sew_resnet(arch, block, layers, c_dim,cdim_cur,pretrained, progress, cnf,  **kwargs):
    model = SEWResNet(block, layers, c_dim=c_dim,cdim_cur=cdim_cur,cnf=cnf,  **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
 
@register_model
def sew_resnet18(c_dim=[64,128,256,512],cdim_cur=[],pretrained=False, progress=True, cnf: str = None,  **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param node: a spiking neuron layer
    :type node: callable
    :param kwargs: kwargs for `node`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module
    The spike-element-wise ResNet-18 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_ modified by the ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _sew_resnet('resnet18', BasicBlock, [2, 2, 2, 2], c_dim,cdim_cur,pretrained, progress,  'ADD', **kwargs)

