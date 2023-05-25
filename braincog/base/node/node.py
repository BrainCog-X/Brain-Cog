# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/4/10 18:46
# User      : Floyed
# Product   : PyCharm
# Project   : braincog
# File      : node.py
# explain   : 神经元节点类型

import abc
import math
from abc import ABC
import numpy as np
import random
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from einops import rearrange, repeat
from braincog.base.strategy.surrogate import *


class BaseNode(nn.Module, abc.ABC):
    """
    神经元模型的基类
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param mem_detach: 是否将上一时刻的膜电位在计算图中截断
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=.5,
                 v_reset=0.,
                 dt=1.,
                 step=8,
                 requires_thres_grad=False,
                 sigmoid_thres=False,
                 requires_fp=False,
                 layer_by_layer=False,
                 n_groups=1,
                 *args,
                 **kwargs):

        super(BaseNode, self).__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=requires_thres_grad)
        self.sigmoid_thres = sigmoid_thres
        self.mem = 0.
        self.spike = 0.
        self.dt = dt
        self.feature_map = []
        self.mem_collect = []
        self.requires_fp = requires_fp
        self.v_reset = v_reset
        self.step = step
        self.layer_by_layer = layer_by_layer
        self.groups = n_groups
        self.mem_detach = kwargs['mem_detach'] if 'mem_detach' in kwargs else False
        self.requires_mem = kwargs['requires_mem'] if 'requires_mem' in kwargs else False

    @abc.abstractmethod
    def calc_spike(self):
        """
        通过当前的mem计算是否发放脉冲，并reset
        :return: None
        """

        pass

    def integral(self, inputs):
        """
        计算由当前inputs对于膜电势的累积
        :param inputs: 当前突触输入电流
        :type inputs: torch.tensor
        :return: None
        """

        pass

    def get_thres(self):
        return self.threshold if not self.sigmoid_thres else self.threshold.sigmoid()

    def rearrange2node(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, 'b (c t) w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, 'b (c t) -> t b c', t=self.step)
            else:
                raise NotImplementedError

        elif self.layer_by_layer:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, '(t b) c w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, '(t b) c -> t b c', t=self.step)
            else:
                raise NotImplementedError


        else:
            outputs = inputs

        return outputs

    def rearrange2op(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> b (c t) w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> b (c t)')
            else:
                raise NotImplementedError
        elif self.layer_by_layer:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> (t b) c w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> (t b) c')
            else:
                raise NotImplementedError

        else:
            outputs = inputs

        return outputs

    def forward(self, inputs):
        """
        torch.nn.Module 默认调用的函数，用于计算膜电位的输入和脉冲的输出
        在```self.requires_fp is True``` 的情况下，可以使得```self.feature_map```用于记录trace
        :param inputs: 当前输入的膜电位
        :return: 输出的脉冲
        """

        if self.layer_by_layer or self.groups != 1:
            inputs = self.rearrange2node(inputs)

            outputs = []
            for i in range(self.step):
                
                if self.mem_detach and hasattr(self.mem, 'detach'):
                    self.mem = self.mem.detach()
                    self.spike = self.spike.detach()
                self.integral(inputs[i])
                
                self.calc_spike()
                
                if self.requires_fp is True:
                    self.feature_map.append(self.spike)
                if self.requires_mem is True:
                    self.mem_collect.append(self.mem)
                outputs.append(self.spike)
            outputs = torch.stack(outputs)

            outputs = self.rearrange2op(outputs)
            return outputs
        else:
            if self.mem_detach and hasattr(self.mem, 'detach'):
                self.mem = self.mem.detach()
                self.spike = self.spike.detach()
            self.integral(inputs)
            self.calc_spike()
            if self.requires_fp is True:
                self.feature_map.append(self.spike)
            if self.requires_mem is True:
                self.mem_collect.append(self.mem)   
            return self.spike

    def n_reset(self):
        """
        神经元重置，用于模型接受两个不相关输入之间，重置神经元所有的状态
        :return: None
        """
        self.mem = self.v_reset
        self.spike = 0.
        self.feature_map = []
        self.mem_collect = []
    def get_n_attr(self, attr):

        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return None

    def set_n_warm_up(self, flag):
        """
        一些训练策略会在初始的一些epoch，将神经元视作ANN的激活函数训练，此为设置是否使用该方法训练
        :param flag: True：神经元变为激活函数， False：不变
        :return: None
        """
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        """
        动态设置神经元的阈值
        :param thresh: 阈值
        :return:
        """
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)

    def set_n_tau(self, tau):
        """
        动态设置神经元的衰减系数，用于带Leaky的神经元
        :param tau: 衰减系数
        :return:
        """
        if hasattr(self, 'tau'):
            self.tau = Parameter(torch.tensor(tau, dtype=torch.float), requires_grad=False)
        else:
            raise NotImplementedError

#============================================================================
# node的基类
class BaseMCNode(nn.Module, abc.ABC):
    """
    多房室神经元模型的基类
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param comps: 神经元不同房室, 例如["apical", "basal", "soma"]
    """
    def __init__(self,
                 threshold=1.0,
                 v_reset=0.,
                 comps=[]):
        super().__init__()
        self.threshold = Parameter(torch.tensor(threshold), requires_grad=False)
        # self.decay = Parameter(torch.tensor(decay), requires_grad=False)
        self.v_reset = v_reset
        assert len(comps) != 0
        self.mems = dict()
        for c in comps:
            self.mems[c] = None 
        self.spike = None
        self.warm_up = False

    @abc.abstractmethod
    def calc_spike(self):
        pass
    @abc.abstractmethod
    def integral(self, inputs):
        pass        
    
    def forward(self, inputs: dict):
        '''
        Params:
            inputs dict: Inputs for every compartments of neuron 
        '''
        if self.warm_up:
            return inputs
        else:
            self.integral(**inputs)
            self.calc_spike()
            return self.spike

    def n_reset(self):
        for c in self.mems.keys():
            self.mems[c] = self.v_reset
        self.spike = 0.0

    def get_n_fire_rate(self):
        if self.spike is None:
            return 0.
        return float((self.spike.detach() >= self.threshold).sum()) / float(np.product(self.spike.shape))

    def set_n_warm_up(self, flag):
        self.warm_up = flag

    def set_n_threshold(self, thresh):
        self.threshold = Parameter(torch.tensor(thresh, dtype=torch.float), requires_grad=False)


class ThreeCompNode(BaseMCNode):
    """
    三房室神经元模型
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param tau: 胞体膜电位时间常数, 用于控制胞体膜电位衰减
    :param tau_basal: 基底树突膜电位时间常数, 用于控制基地树突胞体膜电位衰减
    :param tau_apical: 远端树突膜电位时间常数, 用于控制远端树突胞体膜电位衰减
    :param comps: 神经元不同房室, 例如["apical", "basal", "soma"]
    :param act_fun: 脉冲梯度代理函数
    """
    def __init__(self,
                 threshold=1.0,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 v_reset=0.0,
                 comps=['basal', 'apical', 'soma'],
                 act_fun=AtanGrad):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold, v_reset, comps)
        self.tau = tau
        self.tau_basal = tau_basal
        self.tau_apical = tau_apical
        self.act_fun = act_fun(alpha=tau, requires_grad=False)
    
    def integral(self, basal_inputs, apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''

        self.mems['basal'] =  (self.mems['basal'] + basal_inputs) / self.tau_basal
        self.mems['apical'] =  (self.mems['apical'] + apical_inputs) / self.tau_apical

        self.mems['soma'] = self.mems['soma'] + (self.mems['apical'] + self.mems['basal'] - self.mems['soma']) / self.tau


    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['basal'] = self.mems['basal'] * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())


#============================================================================

# 用于静态测试 使用ANN的情况 不累积电位 
class ReLUNode(BaseNode):
    """
    用于相同连接的ANN的测试
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(requires_fp=False, *args, **kwargs)
        self.act_fun = nn.ReLU()

    def forward(self, x):
        """
        参考```BaseNode```
        :param x:
        :return:
        """
        self.spike = self.act_fun(x)
        if self.requires_fp is True:
            self.feature_map.append(self.spike)
        if self.requires_mem is True:
            self.mem_collect.append(self.mem)
        return self.spike

    def calc_spike(self):
        pass


class BiasReLUNode(BaseNode):
    """
    用于相同连接的ANN的测试, 会在每个时刻注入恒定电流, 使得神经元更容易激发
    """

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.act_fun = nn.ReLU()

    def forward(self, x):
        self.spike = self.act_fun(x + 0.1)
        if self.requires_fp is True:
            self.feature_map += self.spike
        return self.spike

    def calc_spike(self):
        pass


# ============================================================================
# 用于SNN的node
class IFNode(BaseNode):
    """
    Integrate and Fire Neuron
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param requires_thres_grad: 是否需要计算对于threshold的梯度, 默认为 ``False``
    :param sigmoid_thres: 是否使用sigmoid约束threshold的范围搭到 [0, 1], 默认为 ``False``
    :param requires_fp: 是否需要在推理过程中保存feature map, 需要消耗额外的内存和时间, 默认为 ``False``
    :param layer_by_layer: 是否以一次性计算所有step的输出, 在网络模型较大的情况下, 一般会缩短单次推理的时间, 默认为 ``False``
    :param n_groups: 在不同的时间步, 是否使用不同的权重, 默认为 ``1``, 即不分组
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=.5, act_fun=AtanGrad, *args, **kwargs):
        """
        :param threshold:
        :param act_fun:
        :param args:
        :param kwargs:
        """
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)

    def integral(self, inputs):
        self.mem = self.mem + inputs * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class LIFNode(BaseNode):
    """
    Leaky Integrate and Fire
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

    def __init__(self, threshold=0.5, tau=2., act_fun=QGateGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        # self.threshold = threshold
        # print(threshold)
        # print(tau)

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())


class BurstLIFNode(LIFNode):
    def __init__(self, threshold=.5, tau=2., act_fun=RoundGrad, *args, **kwargs):
        super().__init__(threshold=threshold, tau=tau, act_fun=act_fun, *args, **kwargs)
        self.burst_factor = 1.5

    def calc_spike(self):
        LIFNode.calc_spike(self)
        self.spike = torch.where(self.spike > 1., self.burst_factor * self.spike, self.spike)



class BackEINode(BaseNode):
    """
    BackEINode with self feedback connection and excitatory and inhibitory neurons
    Reference：https://www.sciencedirect.com/science/article/pii/S0893608022002520
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param if_back whether to use self feedback
    :param if_ei whether to use excitotory and inhibitory neurons
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """
    def __init__(self, threshold=0.5, decay=0.2, act_fun=BackEIGateGrad, th_fun=EIGrad, channel=40, if_back=True,
                 if_ei=True, cfg_backei=2, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.decay = decay
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        if isinstance(th_fun, str):
            th_fun = eval(th_fun)
        self.act_fun = act_fun()
        self.th_fun = th_fun()
        self.channel = channel
        self.if_back = if_back

        if self.if_back:
            self.back = nn.Conv2d(channel, channel, kernel_size=2 * cfg_backei+1, stride=1, padding=cfg_backei)
        self.if_ei = if_ei
        if self.if_ei:
            self.ei = nn.Conv2d(channel, channel, kernel_size=2 * cfg_backei+1, stride=1, padding=cfg_backei)

    def integral(self, inputs):
        if self.mem is None:
            self.mem = torch.zeros_like(inputs)
            self.spike = torch.zeros_like(inputs)
        self.mem = self.decay * self.mem
        if self.if_back:
            self.mem += F.sigmoid(self.back(self.spike)) * inputs
        else:
            self.mem += inputs

    def calc_spike(self):
        if self.if_ei:
            ei_gate = self.th_fun(self.ei(self.mem))
            self.spike = self.act_fun(self.mem-self.threshold)
            self.mem = self.mem * (1 - self.spike)
            self.spike = ei_gate * self.spike
        else:
            self.spike = self.act_fun(self.mem-self.threshold)
            self.mem = self.mem * (1 - self.spike)

    def n_reset(self):
        self.mem = None
        self.spike = None
        self.feature_map = []
        self.mem_collect = []


class NoiseLIFNode(LIFNode):
    """
    Noisy Leaky Integrate and Fire
    在神经元中注入噪声, 默认的噪声分布为 ``Beta(log(2), log(6))``
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
    :param log_alpha: 控制 beta 分布的参数 ``a``
    :param log_beta: 控制 beta 分布的参数 ``b``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self,
                 threshold=1,
                 tau=2.,
                 act_fun=GateGrad,
                 log_alpha=np.log(2),
                 log_beta=np.log(6),
                 *args,
                 **kwargs):
        super().__init__(threshold=threshold, tau=tau, act_fun=act_fun, *args, **kwargs)
        self.log_alpha = Parameter(torch.as_tensor(log_alpha), requires_grad=True)
        self.log_beta = Parameter(torch.as_tensor(log_beta), requires_grad=True)

        # self.fc = nn.Sequential(
        #     nn.Linear(1, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 2)
        # )

    def integral(self, inputs):  # b, c, w, h / b, c
        # self.mu, self.log_var = self.fc(inputs.mean().unsqueeze(0)).split(1)
        alpha, beta = torch.exp(self.log_alpha), torch.exp(self.log_beta)
        mu = alpha / (alpha + beta)
        var = ((alpha + 1) * alpha) / ((alpha + beta + 1) * (alpha + beta))
        noise = torch.distributions.beta.Beta(alpha, beta).sample(inputs.shape) * self.get_thres()
        noise = noise * var / var.detach() + mu - mu.detach()

        self.mem = self.mem + ((inputs - self.mem) / self.tau + noise) * self.dt


class BiasLIFNode(BaseNode):
    """
    带有恒定电流输入Bias的LIF神经元，用于带有抑制性/反馈链接的网络的测试
    Noisy Leaky Integrate and Fire
    在神经元中注入噪声, 默认的噪声分布为 ``Beta(log(2), log(6))``
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

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)

    def integral(self, inputs):
        self.mem = self.mem + ((inputs - self.mem) / self.tau) * self.dt + 0.1

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class LIFSTDPNode(BaseNode):
    """
    用于执行STDP运算时使用的节点 decay的方式是膜电位乘以decay并直接加上输入电流
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)

    def integral(self, inputs):
        self.mem = self.mem * self.tau + inputs

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        # print(( self.threshold).max())
        self.mem = self.mem * (1 - self.spike.detach())

    def requires_activation(self):
        return False


class PLIFNode(BaseNode):
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

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        init_w = -math.log(tau - 1.)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def integral(self, inputs):
        self.mem = self.mem + ((inputs - self.mem) * self.w.sigmoid()) * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class NoisePLIFNode(PLIFNode):
    """
    Noisy Parametric Leaky Integrate and Fire
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

    def __init__(self,
                 threshold=1,
                 tau=2.,
                 act_fun=GateGrad,
                 *args,
                 **kwargs):
        super().__init__(threshold=threshold, tau=tau, act_fun=act_fun, *args, **kwargs)
        log_alpha = kwargs['log_alpha'] if 'log_alpha' in kwargs else np.log(2)
        log_beta = kwargs['log_beta'] if 'log_beta' in kwargs else np.log(6)
        self.log_alpha = Parameter(torch.as_tensor(log_alpha), requires_grad=True)
        self.log_beta = Parameter(torch.as_tensor(log_beta), requires_grad=True)

        # self.fc = nn.Sequential(
        #     nn.Linear(1, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 5),
        #     nn.ReLU(),
        #     nn.Linear(5, 2)
        # )

    def integral(self, inputs):  # b, c, w, h / b, c
        # self.mu, self.log_var = self.fc(inputs.mean().unsqueeze(0)).split(1)
        alpha, beta = torch.exp(self.log_alpha), torch.exp(self.log_beta)
        mu = alpha / (alpha + beta)
        var = ((alpha + 1) * alpha) / ((alpha + beta + 1) * (alpha + beta))
        noise = torch.distributions.beta.Beta(alpha, beta).sample(inputs.shape) * self.get_thres()
        noise = noise * var / var.detach() + mu - mu.detach()
        self.mem = self.mem + ((inputs - self.mem) * self.w.sigmoid() + noise) * self.dt


class BiasPLIFNode(BaseNode):
    """
    Parametric LIF with bias
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

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        init_w = -math.log(tau - 1.)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def integral(self, inputs):
        self.mem = self.mem + ((inputs - self.mem) * self.w.sigmoid() + 0.1) * self.dt

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())
        self.mem = self.mem * (1 - self.spike.detach())


class DoubleSidePLIFNode(LIFNode):
    """
    能够输入正负脉冲的 PLIF
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

    def __init__(self,
                 threshold=.5,
                 tau=2.,
                 act_fun=AtanGrad,
                 *args,
                 **kwargs):
        super().__init__(threshold, tau, act_fun, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=True)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres()) - self.act_fun(self.get_thres - self.mem)
        self.mem = self.mem * (1. - torch.abs(self.spike.detach()))


class IzhNode(BaseNode):
    """
    Izhikevich 脉冲神经元
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.a = kwargs['a'] if 'a' in kwargs else 0.02
        self.b = kwargs['b'] if 'b' in kwargs else 0.2
        self.c = kwargs['c'] if 'c' in kwargs else -55.
        self.d = kwargs['d'] if 'd' in kwargs else -2.
        '''
        v' = 0.04v^2 + 5v + 140 -u + I
        u' = a(bv-u)
        下面是将Izh离散化的写法
        if v>= thresh:
            v = c
            u = u + d
        '''
        # 初始化膜电势 以及 对应的U
        self.mem = 0.
        self.u = 0.
        self.dt = kwargs['dt'] if 'dt' in kwargs else 1.

    def integral(self, inputs):
        self.mem = self.mem + self.dt * (0.04 * self.mem * self.mem + 5 * self.mem - self.u + 140 + inputs)
        self.u = self.u + self.dt * (self.a * self.b * self.mem - self.a * self.u)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.get_thres())  # 大于阈值释放脉冲
        self.mem = self.mem * (1 - self.spike.detach()) + self.spike.detach() * self.c
        self.u = self.u + self.spike.detach() * self.d

    def n_reset(self):
        self.mem = 0.
        self.u = 0.
        self.spike = 0.


class IzhNodeMU(BaseNode):
    """
    Izhikevich 脉冲神经元多参数版
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.a = kwargs['a'] if 'a' in kwargs else 0.02
        self.b = kwargs['b'] if 'b' in kwargs else 0.2
        self.c = kwargs['c'] if 'c' in kwargs else -55.
        self.d = kwargs['d'] if 'd' in kwargs else -2.
        self.mem = kwargs['mem'] if 'mem' in kwargs else 0.
        self.u = kwargs['u'] if 'u' in kwargs else 0.
        self.dt = kwargs['dt'] if 'dt' in kwargs else 1.

    def integral(self, inputs):
        self.mem = self.mem + self.dt * (0.04 * self.mem * self.mem + 5 * self.mem - self.u + 140 + inputs)
        self.u = self.u + self.dt * (self.a * self.b * self.mem - self.a * self.u)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach()) + self.spike.detach() * self.c
        self.u = self.u + self.spike.detach() * self.d

    def n_reset(self):
        self.mem = -70.
        self.u = 0.
        self.spike = 0.

    def requires_activation(self):
        return False


class DGLIFNode(BaseNode):
    """
    Reference: https://arxiv.org/abs/2110.08858
    :param threshold: 神经元的脉冲发放阈值
    :param tau: 神经元的膜常数, 控制膜电位衰减
    """

    def __init__(self, threshold=.5, tau=2., *args, **kwargs):
        super().__init__(threshold, tau, *args, **kwargs)
        self.act = nn.ReLU()
        self.tau = tau

    def integral(self, inputs):
        inputs = self.act(inputs)
        self.mem = self.mem + ((inputs - self.mem) / self.tau) * self.dt

    def calc_spike(self):
        spike = self.mem.clone()
        spike[(spike < self.get_thres())] = 0.
        # self.spike = spike / (self.mem.detach().clone() + 1e-12)
        self.spike = spike - spike.detach() + \
                     torch.where(spike.detach() > self.get_thres(), torch.ones_like(spike), torch.zeros_like(spike))
        self.spike = spike
        self.mem = torch.where(self.mem >= self.get_thres(), torch.zeros_like(self.mem), self.mem)


class HTDGLIFNode(IFNode):
    """
    Reference: https://arxiv.org/abs/2110.08858
    :param threshold: 神经元的脉冲发放阈值
    :param tau: 神经元的膜常数, 控制膜电位衰减
    """

    def __init__(self, threshold=.5, tau=2., *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.warm_up = False

    def calc_spike(self):
        spike = self.mem.clone()
        spike[(spike < self.get_thres())] = 0.
        # self.spike = spike / (self.mem.detach().clone() + 1e-12)
        self.spike = spike - spike.detach() + \
                     torch.where(spike.detach() > self.get_thres(), torch.ones_like(spike), torch.zeros_like(spike))
        self.spike = spike
        self.mem = torch.where(self.mem >= self.get_thres(), torch.zeros_like(self.mem), self.mem)
        # self.mem[[(spike > self.get_thres())]] = self.mem[[(spike > self.get_thres())]] - self.get_thres()

        self.mem = (self.mem + 0.2 * self.spike - 0.2 * self.spike.detach()) * self.dt

    def forward(self, inputs):
        if self.warm_up:
            return F.relu(inputs)
        else:
            return super(IFNode, self).forward(F.relu(inputs))


class SimHHNode(BaseNode):
    """
    简单版本的HH模型
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, threshold=50., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        '''
        I = Cm dV/dt + g_k*n^4*(V_m-V_k) + g_Na*m^3*h*(V_m-V_Na) + g_l*(V_m - V_L)
        '''
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.g_Na, self.g_K, self.g_l = torch.tensor(120.), torch.tensor(120), torch.tensor(0.3)  # k 36
        self.V_Na, self.V_K, self.V_l = torch.tensor(120.), torch.tensor(-120.), torch.tensor(10.6)  # k -12
        self.m, self.n, self.h = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        self.mem = 0
        self.dt = 0.01

    def integral(self, inputs):
        self.I_Na = torch.pow(self.m, 3) * self.g_Na * self.h * (self.mem - self.V_Na)
        self.I_K = torch.pow(self.n, 4) * self.g_K * (self.mem - self.V_K)
        self.I_L = self.g_l * (self.mem - self.V_l)
        self.mem = self.mem + self.dt * (inputs - self.I_Na - self.I_K - self.I_L) / 0.02
        # non Na
        # self.mem = self.mem + 0.01 * (inputs -  self.I_K - self.I_L) / 0.02  #decayed
        # NON k
        # self.mem = self.mem + 0.01 * (inputs - self.I_Na - self.I_L) / 0.02  #increase

        self.alpha_n = 0.01 * (self.mem + 10.0) / (1 - torch.exp(-(self.mem + 10.0) / 10))
        self.beta_n = 0.125 * torch.exp(-(self.mem) / 80)

        self.alpha_m = 0.1 * (self.mem + 25) / (1 - torch.exp(-(self.mem + 25) / 10))
        self.beta_m = 4 * torch.exp(-(self.mem) / 18)

        self.alpha_h = 0.07 * torch.exp(-(self.mem) / 20)
        self.beta_h = 1 / (1 + torch.exp(-(self.mem + 30) / 10))

        self.n = self.n + self.dt * (self.alpha_n * (1 - self.n) - self.beta_n * self.n)
        self.m = self.m + self.dt * (self.alpha_m * (1 - self.m) - self.beta_m * self.m)
        self.h = self.h + self.dt * (self.alpha_h * (1 - self.h) - self.beta_h * self.h)

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike

    def n_reset(self):
        self.mem = 0.
        self.spike = 0.
        self.m, self.n, self.h = torch.tensor(0), torch.tensor(0), torch.tensor(0)

    def requires_activation(self):
        return False


class CTIzhNode(IzhNode):
    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold, tau, act_fun, *args, **kwargs)

        self.name = kwargs['name'] if 'name' in kwargs else ''
        self.excitability = kwargs['excitability'] if 'excitability' in kwargs else 'TRUE'
        self.spikepattern = kwargs['spikepattern'] if 'spikepattern' in kwargs else 'RS'
        self.synnum = kwargs['synnum'] if 'synnum' in kwargs else 0
        self.locationlayer = kwargs['locationlayer'] if 'locationlayer' in kwargs else ''
        self.adjneuronlist = {}
        self.proximal_dendrites = []
        self.distal_dendrites = []
        self.totalindex = kwargs['totalindex'] if 'totalindex' in kwargs else 0
        self.colindex = 0
        self.state = 'inactive'

        self.Gup = kwargs['Gup'] if 'Gup' in kwargs else 0.0
        self.Gdown = kwargs['Gdown'] if 'Gdown' in kwargs else 0.0
        self.Vr = kwargs['Vr'] if 'Vr' in kwargs else 0.0
        self.Vt = kwargs['Vt'] if 'Vt' in kwargs else 0.0
        self.Vpeak = kwargs['Vpeak'] if 'Vpeak' in kwargs else 0.0
        self.capicitance = kwargs['capacitance'] if 'capacitance' in kwargs else 0.0
        self.k = kwargs['k'] if 'k' in kwargs else 0.0
        self.mem = -65
        self.vtmp = -65
        self.u = -13.0
        self.spike = 0
        self.dc = 0

    def integral(self, inputs):
        self.mem += self.dt * (
                self.k * (self.mem - self.Vr) * (self.mem - self.Vt) - self.u + inputs) / self.capicitance
        self.u += self.dt * (self.a * (self.b * (self.mem - self.Vr) - self.u))

    def calc_spike(self):
        if self.mem >= self.Vpeak:
            self.mem = self.c
            self.u = self.u + self.d
            self.spike = 1
            self.spreadMarkPostNeurons()

    def spreadMarkPostNeurons(self):
        for post, list in self.adjneuronlist.items():
            if self.excitability == "TRUE":
                post.dc = random.randint(140, 160)
            else:
                post.dc = random.randint(-160, -140)


class adth(BaseNode):
    """
        The adaptive Exponential Integrate-and-Fire model (aEIF)
        :param args: Other parameters
        :param kwargs: Other parameters
    """

    def __init__(self, *args, **kwargs):
        super().__init__(requires_fp=False, *args, **kwargs)

    def adthNode(self, v, dt, c_m, g_m, alpha_w, ad, Ieff, Ichem, Igap, tau_ad, beta_ad, vt, vm1):
        """
                Calculate the neurons that discharge after the current threshold is reached
                :param v: Current neuron voltage
                :param dt: time step
                :param ad:Adaptive variable
                :param vv:Spike, if the voltage exceeds the threshold from below
        """
        v = v + dt / c_m * (-g_m * v + alpha_w * ad + Ieff + Ichem + Igap)
        ad = ad + dt / tau_ad * (-ad + beta_ad * v)
        vv = (v >= vt).astype(int) * (vm1 < vt).astype(int)
        vm1 = v
        return v, ad, vv, vm1

    def calc_spike(self):
        pass
    
    
class HHNode(BaseNode):
    """
    用于脑模拟的HH模型
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param dt: 时间步长
    :param step: 仿真步
    :param tau: 膜电位时间常数, 用于控制膜电位衰减
    :param act_fun: 使用surrogate gradient 对梯度进行近似, 默认为 ``surrogate.AtanGrad``
    :param args: 其他的参数
    :param kwargs: 其他的参数
    """

    def __init__(self, p, neuron_num, W, type_index, act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold=p[7], *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        '''
        I = Cm dV/dt + g_k*n^4*(V_m-V_k) + g_Na*m^3*h*(V_m-V_Na) + g_l*(V_m - V_L)
        '''
        self.neuron_num = neuron_num
        self.type_index = type_index
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.tau_I = 1
        self.g_Na = torch.tensor(p[0])
        self.g_K = torch.tensor(p[1])
        self.g_l = torch.tensor(p[2])
        self.V_Na = torch.tensor(p[3])
        self.V_K = torch.tensor(p[4])
        self.V_l = torch.tensor(p[5])
        self.C = torch.tensor(p[6])
        self.m = torch.zeros(neuron_num)
        self.n = torch.zeros(neuron_num)
        self.h = torch.zeros(neuron_num)
        self.v_reset = 0
        self.W = W
        self.if_IN = p[8]
        self.dt_over_tau = self.dt / self.tau_I
        self.sqrt_coeff = 1
        if self.if_IN:
            self.mu = -3
        else:
            self.mu = 3
        self.sig = 12
        self.Gama_c = 1/9
        self.mem = self.threshold * torch.rand((1, neuron_num)).squeeze(0)
        # self.mem = torch.zeros(neuron_num)
        self.spike = torch.zeros(neuron_num)
        self.dt = 0.1
        self.Iback = torch.zeros(neuron_num)
        self.Ieff = torch.zeros(neuron_num)
        self.Ichem = torch.zeros(neuron_num)
        if self.if_IN:
            self.Igap = torch.zeros(neuron_num)

    def integral(self, inputs):
        self.I_Na = torch.pow(self.m, 3) * self.g_Na * self.h * (self.mem - self.V_Na)
        self.I_K = torch.pow(self.n, 4) * self.g_K * (self.mem - self.V_K)
        self.I_L = self.g_l * (self.mem - self.V_l)
        # print(self.I_Na)
        self.mem = self.mem + self.dt * (inputs - self.I_Na - self.I_K - self.I_L) / self.C

        self.alpha_n = 0.01 * (self.mem + 10.0) / (1 - torch.exp(-(self.mem + 10.0) / 10))
        self.beta_n = 0.125 * torch.exp(-(self.mem) / 80)

        self.alpha_m = 0.1 * (self.mem + 25) / (1 - torch.exp(-(self.mem + 25) / 10))
        self.beta_m = 4 * torch.exp(-(self.mem) / 18)

        self.alpha_h = 0.07 * torch.exp(-(self.mem) / 20)
        self.beta_h = 1 / (1 + torch.exp(-(self.mem + 30) / 10))

        self.n = self.n + self.dt * (self.alpha_n * (1 - self.n) - self.beta_n * self.n) / 10
        self.m = self.m + self.dt * (self.alpha_m * (1 - self.m) - self.beta_m * self.m) / 10
        self.h = self.h + self.dt * (self.alpha_h * (1 - self.h) - self.beta_h * self.h) / 10

    def calc_spike(self):
        self.spike = (self.mem > self.threshold).float()
        self.mem = self.mem * (1 - self.spike.detach())
        # print(self.mem[0])

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike, self.mem

    def n_reset(self):
        self.mem = 0.
        self.spike = 0.
        self.m, self.n, self.h = torch.tensor(0), torch.tensor(0), torch.tensor(0)

    def requires_activation(self):
        return False

    
    
class aEIF(BaseNode):
    """
        The adaptive Exponential Integrate-and-Fire model (aEIF)
        This class define the membrane, spike, current and parameters of a neuron group of a specific type
        :param args: Other parameters
        :param kwargs: Other parameters
    """

    def __init__(self, p, neuron_num, W, type_index, *args, **kwargs):
        """
            p:[threshold, c_m, alpha_w, beta_ad, mu, sig, if_IN]
            c_m: Membrane capacitance
            alpha_w: Coupling of the adaptation variable
            beta_ad: Conductance of the adaptation variable
            mu: Mean of back current
            sig: Variance of back current
            if_IN: if the neuron type is inhibitory neuron, it has gap-junction

            neuron_num: number of neurons in this group
            W: connection weight for the neuron groups connected to this group
            type_index: the index of this type of neuron group in the brain region
        """
        super().__init__(threshold=p[0], requires_fp=False, *args, **kwargs)
        self.neuron_num = neuron_num
        self.type_index = type_index
        self.g_m = 1  # neuron conduction
        self.dt = 1
        self.tau_I = 3  # Time constant to filter the synaptic inputs
        self.tau_ad = 6  # Time constant of adaption coupling
        self.Delta_T = 0.5  # parameter
        gamma_c = 0.1
        self.Gama_c = self.g_m * gamma_c / (1 - gamma_c)  # gap-junction parameter
        self.v_reset = -15  # membrane potential reset to v_reset after fire spike
        self.if_IN = p[6]
        self.beta_ad = p[3]
        if self.if_IN:
            self.c_m = p[1] * (self.g_m + self.Gama_c)  # neuron membrane time constant
            self.alpha_w = p[2] * (self.g_m + self.Gama_c)  # effective adaption coupling
            self.mu = p[4] * (self.g_m + self.Gama_c)
            self.sig = p[5] * (self.g_m + self.Gama_c)  # effective mean and variance of back current
        else:
            self.c_m = p[1] * self.g_m
            self.alpha_w = p[2] * self.g_m
            self.mu = p[4] * self.g_m
            self.sig = p[5] * self.g_m
        self.W = W * self.c_m / self.dt
        self.refrac = 5 / self.dt  # refractory period
        self.dt_over_tau = self.dt / self.tau_I
        self.sqrt_coeff = math.sqrt(1 / (2 * (1 / self.dt_over_tau)))
        self.mem = self.v_reset * torch.ones(neuron_num)
        self.spike = torch.zeros(neuron_num)
        self.ad = torch.zeros(neuron_num)
        self.ref = torch.randint(0, int(self.refrac + 1), (1, neuron_num)).squeeze(0)  # refractory counter
        self.Iback = torch.zeros(neuron_num)
        self.Ieff = torch.zeros(neuron_num)
        self.Ichem = torch.zeros(neuron_num)
        if self.if_IN:
            self.Igap = torch.zeros(neuron_num)

    def integral(self, inputs):

        self.mem = self.mem + (self.ref > self.refrac) * self.dt / self.c_m * \
                   (-self.g_m * (self.mem - self.v_reset) + self.g_m * self.Delta_T *
                    torch.exp((self.mem - self.threshold) / self.Delta_T) +
                    self.alpha_w * self.ad + inputs)

        self.ad = self.ad + (self.ref > self.refrac) * self.dt / self.tau_ad * \
                  (-self.ad + self.beta_ad * (self.mem - self.v_reset))

    def calc_spike(self):
        self.spike = (self.mem > self.threshold).float()
        self.ref = self.ref * (1 - self.spike) + 1
        self.ad = self.ad + self.spike * 30
        self.mem = self.spike * self.v_reset + (1 - self.spike.detach()) * self.mem

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike, self.mem
    
class LIAFNode(BaseNode):
    """
    Leaky Integrate and Analog Fire (LIAF), Reference: https://ieeexplore.ieee.org/abstract/document/9429228
    与LIF相同, 但前传的是膜电势, 更新沿用阈值和膜电势
    :param act_fun: 前传使用的激活函数 [ReLU, SeLU, LeakyReLU]
    :param threshold_related: 阈值依赖模式，若为"True"则 self.spike = act_fun(mem-threshold)
    :note that BaseNode return self.spike, and here self.spike is analog value.
    """
    def __init__(self, spike_act=BackEIGateGrad(), act_fun="SELU", threshold=0.5, tau=2., threshold_related=True, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval("nn." + act_fun + "()")
        self.tau = tau
        self.act_fun = act_fun
        self.spike_act = spike_act
        self.threshold_related = threshold_related

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        if self.threshold_related:
            spike_tmp = self.act_fun(self.mem - self.threshold)
        else:
            spike_tmp = self.act_fun(self.mem)
        self.spike = self.spike_act(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike)
        self.spike = spike_tmp



class OnlineLIFNode(BaseNode):
    """
    Online-update Leaky Integrate and Fire
    与LIF模型相同，但是时序信息在反传时从计算图剥离，因此可以实现在线的更新；模型占用显存固定，不随仿真步step线性提升。
    使用此神经元需要修改:  1. 将模型中t次forward从model_zoo写到main.py中
                       2. 在Conv层与OnelineLIFNode层中加入Replace函数，即时序前传都是detach的，但仍计算该层空间梯度信息。
                       3. 网络结构不适用BN层，使用weight standardization
    注意该神经元不同于OTTT，而是将时序信息全部扔弃。对应这篇文章：https://arxiv.org/abs/2302.14311
    若需保留时序，需要对self.rate_tracking进行计算。实现可参考https://github.com/pkuxmq/OTTT-SNN
    """

    def __init__(self, threshold=0.5, tau=2., act_fun=QGateGrad, init=False, *args, **kwargs):
        super().__init__(threshold, *args, **kwargs)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.rate_tracking = None
        self.init = True


    def integral(self, inputs):
        if self.init is True:
            self.mem = torch.zeros_like(inputs)
            self.init = False
        self.mem = self.mem.detach() + (inputs - self.mem.detach()) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())
        with torch.no_grad():
            if self.rate_tracking == None:
                self.rate_tracking = self.spike.clone().detach()
        self.spike = torch.cat((self.spike, self.rate_tracking), dim=0)