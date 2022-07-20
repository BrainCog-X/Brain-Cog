from functools import partial
from torchvision.ops import DeformConv2d
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *


class BaseLinearModule(nn.Module):
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

    def forward(self, x):
        if self.groups == 1:  # (t b) c
            outputs = self.fc(x)

        else: # b (c t)
            x = rearrange(x, 'b (c t) -> t b c', t=self.groups)
            outputs = []
            for i in range(self.groups):
                outputs.append(self.fc[i](x[i]))
            outputs = torch.stack(outputs) # t b c
            outputs = rearrange(outputs, 't b c -> b (c t)')

        return self.node(outputs)


class BaseConvModule(nn.Module):
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
                 node=PLIFNode,
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
        # self.conv = DeformConvPack(in_channels=in_channels,
        #                            out_channels=out_channels,
        #                            kernel_size=kernel_size,
        #                            padding=padding,
        #                            stride=stride,
        #                            bias=bias)

        self.bn = nn.BatchNorm2d(out_channels * self.groups)

        self.node = partial(node, **kwargs)()

        self.activation = nn.Identity()

    def forward(self, x):
        # origin_shape = x.shape
        # if len(origin_shape) > 4:
        #     x = x.reshape(np.prod(origin_shape[0:-3]), *origin_shape[-3:])
        x = self.conv(x)
        x = self.bn(x)
        # if len(origin_shape) > 4:
        #     x = x.reshape(*origin_shape[0:-3], *x.shape[-3:])

        x = self.node(x)
        return x


class BaseModule(nn.Module, abc.ABC):
    """
    SNN抽象类, 所有的SNN都要继承这个类, 以实现一些基础方法
    :param step: 仿真步长
    :param encode_type: 数据编码类型
    :param layer_by_layer: 是否layer wise地进行前向推理
    :param temporal_flatten: 是否将时间维度和channel合并
    :param args:
    :param kwargs:
    """
    def __init__(self,
                 step,
                 encode_type,
                 layer_by_layer=False,
                 temporal_flatten=False,
                 *args,
                 **kwargs):
        super(BaseModule, self).__init__()
        self.step = step
        # print(kwargs['layer_by_layer'])
        self.layer_by_layer = layer_by_layer

        self.temporal_flatten = temporal_flatten
        encode_step = self.step

        if temporal_flatten is True:
            self.init_channel_mul = self.step
            self.step = 1
        else:  # origin
            self.init_channel_mul = 1

        self.encoder = Encoder(encode_step, encode_type, **kwargs)

        self.kwargs = kwargs
        self.warm_up = False

        self.fire_rate = []

    def reset(self):
        """
        重置所有神经元的膜电位
        :return:
        """
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

    def set_attr(self, attr, val):
        """
        设置神经元的属性
        :param attr: 属性名称
        :param val: 设置的属性值
        :return:
        """
        for mod in self.modules():
            if isinstance(mod, BaseNode):
                if hasattr(mod, attr):
                    setattr(mod, attr, val)
                else:
                    ValueError('{} do not has {}'.format(self, attr))

    def get_threshold(self):
        """
        获取所有神经元的阈值
        :return:
        """
        outputs = []
        for mod in self.modules():
            if isinstance(mod, BaseNode):
                thresh = (mod.get_thres())
                outputs.append(thresh)
        return outputs

    def get_fp(self, temporal_info=False):
        """
        获取所有神经元的状态
        :param temporal_info: 是否要读取神经元的时间维度状态, False会把时间维度拍平
        :return: 所有神经元的状态, List
        """
        outputs = []
        for mod in self.modules():
            if isinstance(mod, BaseNode):
                if temporal_info:
                    outputs.append(mod.feature_map)
                else:
                    outputs.append(sum(mod.feature_map) / len(mod.feature_map))
        return outputs

    def get_fire_rate(self, requires_grad=False):
        """
        获取神经元的fire-rate
        :param requires_grad: 是否需要梯度信息, 默认为 ``False`` 会截断梯度
        :return: 所有神经元的fire-rate
        """
        outputs = []
        fp = self.get_attr('feature_map')
        for f in fp:
            if requires_grad is False:
                if len(f) == 0:
                    return torch.tensor([0.])
                outputs.append(((sum(f) / len(f)).detach() > 0.).float().mean())
            else:
                outputs.append(((sum(f) / len(f)) > 0.).float().mean())
        if len(outputs) == 0:
            return torch.tensor([0.])
        return torch.stack(outputs)

    def get_tot_spike(self):
        """
        获取神经元总的脉冲数量
        :return:
        """
        tot_spike = 0
        batch_size = 1
        fp = self.get_attr('feature_map')
        for f in fp:
            if len(f) == 0:
                break
            tot_spike += sum(f).sum()
            batch_size = f[0].shape[0]
        return tot_spike / batch_size

    def get_spike_info(self):
        """
        获取神经元的脉冲信息, 主要用于绘图
        :return:
        """
        spike_feature_list = self.get_fp(temporal_info=True)
        avg, var, spike = [], [], []
        avg_per_step = []
        for spike_feature in spike_feature_list:
            avg_list = []
            for spike_t in spike_feature:
                avg_list.append(float(spike_t.mean()))
            avg_per_step.append(avg_list)

            spike_feature = sum(spike_feature)
            num = np.prod(spike_feature.shape)
            avg.append(float(spike_feature.sum()))
            var.append(float(spike_feature.std()))
            lst = []
            for t in range(self.step + 1):
                lst.append(float((spike_feature == t).sum() / num))
                # lst.append(  # for mem storage
                #     float(torch.logical_and(spike_feature >= 2 * t / (self.step + 1) - 1,
                #                             spike_feature < 2 * (t + 1) / (self.step + 1) - 1).sum() / num))
            spike.append(lst)

        return avg, var, spike, avg_per_step

    def get_attr(self, attr):
        """
        获取神经元的某一属性值
        :param attr: 属性名称
        :return: 对应属性的值, List
        """
        outputs = []
        for mod in self.modules():
            if hasattr(mod, attr):
                outputs.append(getattr(mod, attr))
        return outputs

    @staticmethod
    def forward(self, inputs):
        pass
