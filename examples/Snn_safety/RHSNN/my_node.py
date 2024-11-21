import torch
from braincog.base.node.node import *

class RHLIFNode(BaseNode):
    """
    Parametric LIF， 其中的 ```tau``` 会被backward过程影响
    Reference：https://arxiv.org/abs/2007.05785
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=0.5, tau=0., sigma=1.0, act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        init_w = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)

        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.sigma = nn.Parameter(torch.as_tensor(sigma), requires_grad=False)
        self.w = nn.Parameter(torch.as_tensor(init_w), requires_grad=False)
        self.flag = 0
        self.rd = 0


    def integral(self, inputs):
        self.rd = self.sigma * torch.normal(0., 1., size=(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]), device=inputs.device)
        self.mem = self.rd.sigmoid() * self.mem + (1 - self.rd.sigmoid()) * inputs

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        # self.mem = self.mem - self.spike.detach() * self.threshold
        self.mem = self.mem * (1 - self.spike.detach())

    def n_reset(self):
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
        self.flag = 0

class RHLIFNode2(BaseNode):
    """
    Parametric LIF， 其中的 ```tau``` 会被backward过程影响
    Reference：https://arxiv.org/abs/2007.05785
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=0.5, tau=0., sigma=1.0, act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        init_w = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)

        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.sigma = nn.Parameter(torch.as_tensor(sigma), requires_grad=False)
        self.w = nn.Parameter(torch.as_tensor(init_w), requires_grad=False)
        self.flag = 0
        self.rd = 0
        self.resample = 1


    def integral(self, inputs):
        if self.flag == 0:
            self.rd = self.sigma * torch.normal(0., 1., size=(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]), device=inputs.device)
            self.flag = 1
        self.mem = self.rd.sigmoid() * self.mem + (1 - self.rd.sigmoid()) * inputs

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        # self.mem = self.mem - self.spike.detach() * self.threshold
        self.mem = self.mem * (1 - self.spike.detach())

    def n_reset(self):
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
        if self.resample == 1:
            self.flag = 0
        else:
            self.flag = 1