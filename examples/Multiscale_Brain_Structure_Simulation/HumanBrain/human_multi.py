import time

import numpy as np
import scipy.io as scio
import torch
from torch import nn
from braincog.base.node.node import *
from braincog.base.brainarea.BrainArea import *
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt

device_ids = [0,2,3,4,5,7,8,9]

device = 'cuda:0'



class MultiCompartmentaEIF(BaseNode):
    """
    双房室神经元模型
    :param threshold: 神经元发放脉冲需要达到的阈值
    :param v_reset: 静息电位
    :param tau: 胞体膜电位时间常数, 用于控制胞体膜电位衰减
    :param tau_basal: 基底树突膜电位时间常数, 用于控制基地树突胞体膜电位衰减
    :param tau_apical: 远端树突膜电位时间常数, 用于控制远端树突胞体膜电位衰减
    :param comps: 神经元不同房室, 例如["apical", "soma"]
    :param act_fun: 脉冲梯度代理函数
    """
    def __init__(self,
                 p,
                 dt,
                 tau=2.0,
                 tau_basal=2.0,
                 tau_apical=2.0,
                 act_fun=AtanGrad, *args, **kwargs):
        g_B = 0.6
        g_L = 0.05
        super().__init__(threshold=p[0], *args, **kwargs)
        self.neuron_num = len(p[0])
        self.tau = 2.0
        self.tau_basal = 20.0
        self.tau_apical = 2.0
        self.spike = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.v_reset = p[1]  # membrane potential reset to v_reset after fire spike
         # Initialize membrane potentials
        self.tau_I = 3.0
        self.sig = 12.0
        self.mu = 10.0
        self.dt=dt
        self.dt_over_tau = self.dt / self.tau_I
        self.mems = {}
        self.mems['soma'] = torch.ones(self.neuron_num, device=device) * self.v_reset
        self.mems['apical'] = torch.ones(self.neuron_num, device=device) * self.v_reset
        self.act_fun = act_fun(alpha=self.tau, requires_grad=False)
        self.Iback = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.Ieff = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.sqrt_coeff = math.sqrt(1 / (2 * (1 / self.dt_over_tau)))
        
        
    
    def integral(self,apical_inputs):
        '''
        Params:
            inputs torch.Tensor: Inputs for basal dendrite  
        '''
        self.mems['apical'] =  (self.mems['apical'] + apical_inputs) / self.tau_apical
        self.mems['soma'] = self.mems['soma'] + (self.mems['apical'] - self.mems['soma']) / self.tau


    def calc_spike(self):
        self.spike = self.act_fun(self.mems['soma'] - self.threshold)
        self.mems['soma'] = self.mems['soma']  * (1. - self.spike.detach())
        self.mems['apical'] = self.mems['apical']  * (1. - self.spike.detach())
    def forward(self, inputs):

        # aeifnode_cuda.forward(self.threshold, self.c_m, self.alpha_w, self.beta_ad, inputs, self.ref, self.ad, self.mem, self.spike)
        self.integral(inputs)
        self.calc_spike()

        return self.spike, self.mems['soma']



class aEIF(BaseNode):
    """
        The adaptive Exponential Integrate-and-Fire model (aEIF)
        This class define the membrane, spike, current and parameters of a neuron group of a specific type
        :param args: Other parameters
        :param kwargs: Other parameters
    """

    def __init__(self, p, dt, device, *args, **kwargs):
        """
            p:[threshold, v_reset, c_m, tao_w, alpha_ad, beta_ad]
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
        self.neuron_num = len(p[0])
        self.g_m = 0.1  # neuron conduction
        self.dt = dt
        self.tau_I = 3  # Time constant to filter the synaptic inputs
        self.Delta_T = 0.5  # parameter
        self.v_reset = p[1]  # membrane potential reset to v_reset after fire spike
        self.c_m = p[2]
        self.tau_w = p[3]  # Time constant of adaption coupling
        self.alpha_ad = p[4]
        self.beta_ad = p[5]
        self.refrac = 5 / self.dt  # refractory period
        self.dt_over_tau = self.dt / self.tau_I
        self.sqrt_coeff = math.sqrt(1 / (2 * (1 / self.dt_over_tau)))
        self.mem = self.v_reset
        self.spike = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.ad = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.ref = torch.randint(0, int(self.refrac + 1), (1, self.neuron_num), device=device, requires_grad=False).squeeze(
            0)  # refractory counter
        self.ref = self.ref.float()
        self.mu = 10
        self.sig = 12
        self.Iback = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.Ieff = torch.zeros(self.neuron_num, device=device, requires_grad=False)

    def integral(self, inputs):

        self.mem = self.mem + (self.ref > self.refrac) * self.dt / self.c_m * \
                   (-self.g_m * (self.mem - self.v_reset) + self.g_m * self.Delta_T *
                    torch.exp((self.mem - self.threshold) / self.Delta_T) +
                    self.alpha_ad * self.ad + inputs)

        self.ad = self.ad + (self.ref > self.refrac) * self.dt / self.tau_w * \
                  (-self.ad + self.beta_ad * (self.mem - self.v_reset))

    def calc_spike(self):
        self.spike = (self.mem > self.threshold).float()
        self.ref = self.ref * (1 - self.spike) + 1
        self.ad = self.ad + self.spike * 30
        self.mem = self.spike * self.v_reset + (1 - self.spike.detach()) * self.mem

    def forward(self, inputs):

        # aeifnode_cuda.forward(self.threshold, self.c_m, self.alpha_w, self.beta_ad, inputs, self.ref, self.ad, self.mem, self.spike)
        self.integral(inputs)
        self.calc_spike()

        return self.spike, self.mem

class HHNode(BaseNode):
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

    def __init__(self, p, dt, device, act_fun=AtanGrad, *args, **kwargs):
        super().__init__(threshold=p[0], *args, **kwargs)
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        '''
        I = Cm dV/dt + g_k*n^4*(V_m-V_k) + g_Na*m^3*h*(V_m-V_Na) + g_l*(V_m - V_L)
        '''
        self.neuron_num = len(p[0])
        self.act_fun = act_fun(alpha=2., requires_grad=False)
        self.tau_I = 3
        self.g_Na = torch.tensor(p[1])
        self.g_K = torch.tensor(p[2])
        self.g_l = torch.tensor(p[3])
        self.V_Na = torch.tensor(p[4])
        self.V_K = torch.tensor(p[5])
        self.V_l = torch.tensor(p[6])
        self.C = torch.tensor(p[7])
        self.m = 0.05 * torch.ones(self.neuron_num, device=device, requires_grad=False)
        self.n = 0.31 * torch.ones(self.neuron_num, device=device, requires_grad=False)
        self.h = 0.59 * torch.ones(self.neuron_num, device=device, requires_grad=False)
        self.v_reset = 0
        self.dt = dt
        self.dt_over_tau = self.dt / self.tau_I
        self.sqrt_coeff = math.sqrt(1 / (2 * (1 / self.dt_over_tau)))
        self.mu = 10
        self.sig = 12

        self.mem = torch.tensor(self.v_reset, device=device, requires_grad=False)
        self.mem_p = self.mem
        self.spike = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.Iback = torch.zeros(self.neuron_num, device=device, requires_grad=False)
        self.Ieff = torch.zeros(self.neuron_num, device=device, requires_grad=False)

    def integral(self, inputs):
        self.alpha_n = (0.1 - 0.01 * self.mem) / (torch.exp(1 - 0.1 * self.mem) - 1)
        self.alpha_m = (2.5 - 0.1 * self.mem) / (torch.exp(2.5 - 0.1 * self.mem) - 1)
        self.alpha_h = 0.07 * torch.exp(-self.mem / 20.0)

        self.beta_n = 0.125 * torch.exp(-self.mem / 80.0)
        self.beta_m = 4.0 * torch.exp(-self.mem / 18.0)
        self.beta_h = 1 / (torch.exp(3 - 0.1 * self.mem) + 1)

        self.tau_n = 1.0 / (self.alpha_n + self.beta_n)
        self.inf_n = self.alpha_n * self.tau_n

        self.tau_m = 1.0 / (self.alpha_m + self.beta_m)
        self.inf_m = self.alpha_m * self.tau_m

        self.tau_h = 1.0 / (self.alpha_h + self.beta_h)
        self.inf_h = self.alpha_h * self.tau_h

        self.n = (1 - self.dt / self.tau_n) * self.n + (self.dt / self.tau_n) * self.inf_n
        self.m = (1 - self.dt / self.tau_m) * self.m + (self.dt / self.tau_m) * self.inf_m
        self.h = (1 - self.dt / self.tau_h) * self.h + (self.dt / self.tau_h) * self.inf_h

        # self.n = self.n + self.dt * (self.alpha_n * (1 - self.n) - self.beta_n * self.n)
        # self.m = self.m + self.dt * (self.alpha_m * (1 - self.m) - self.beta_m * self.m)
        # self.h = self.h + self.dt * (self.alpha_h * (1 - self.h) - self.beta_h * self.h)

        self.I_Na = torch.pow(self.m, 3) * self.g_Na * self.h * (self.mem - self.V_Na)
        self.I_K = torch.pow(self.n, 4) * self.g_K * (self.mem - self.V_K)
        self.I_L = self.g_l * (self.mem - self.V_l)

        self.mem_p = self.mem
        self.mem = self.mem + self.dt * (inputs - self.I_Na - self.I_K - self.I_L) / self.C
        # self.mem = self.mem + self.dt * (inputs - self.I_K - self.I_L) / self.C

    def calc_spike(self):
        self.spike = (self.threshold > self.mem_p).float() * (self.mem > self.threshold).float()

    def forward(self, inputs):
        self.integral(inputs)
        self.calc_spike()
        return self.spike, self.mem

    def requires_activation(self):
        return False

class Syn(nn.Module):
    def __init__(self, syn, weight, neuron_num, tao_d, tao_r, dt, device):
        super().__init__()
        self.pre = syn[1]
        self.post = syn[0]
        self.syn_num = len(syn)
        self.w = torch.sparse_coo_tensor(syn.t(), weight,
                                         size=(neuron_num, neuron_num))
        self.tao_d = tao_d
        self.tao_r = tao_r
        self.dt = dt
        self.lamda_d = self.dt / self.tao_d
        self.lamda_r = self.dt / self.tao_r

        self.s = torch.zeros(neuron_num, device=device)
        self.r = torch.zeros(neuron_num, device=device)
        self.dt = dt

    def forward(self, neuron):
        neuron.Iback = neuron.Iback + neuron.dt_over_tau * (
                torch.randn(neuron.neuron_num, device=device, requires_grad=False) - neuron.Iback)
        neuron.Ieff = neuron.Iback / neuron.sqrt_coeff * neuron.sig + neuron.mu
        self.s = self.s + self.lamda_r * (-self.s + 1 / self.tao_d * neuron.spike)
        self.r = self.r - self.lamda_d * self.r + self.dt * self.s
        self.I = torch.sparse.mm(self.w, self.r.unsqueeze(-1)).squeeze() + neuron.Ieff
        return self.I

class brain(nn.Module):
    def __init__(self, syn, weight, neuron_model, p_neuron, dt, device):
        super().__init__()
        if neuron_model == 'HH':
            self.neurons = HHNode(p_neuron, dt, device)
        elif neuron_model == 'aEIF':
            self.neurons = aEIF(p_neuron, dt, device)
        elif neuron_model == 'MultiCompartmentaEIF':
            self.neurons = MultiCompartmentaEIF(p_neuron,dt,device)
        self.neuron_num = len(p_neuron[0])
        self.syns = Syn(syn, weight, self.neuron_num, 3.0, 6.0, dt, device)

    def forward(self, inputs):
        I = self.syns(self.neurons)
        self.neurons(I)


def brain_region(neuron_num):
    region = []
    start = 0
    end = 0
    for i in range(len(neuron_num)):
        end += neuron_num[i].item()
        region.append([start, end])
        start = end
    return torch.tensor(region)

def neuron_type(neuron_num, ratio, regions):
    neuron_num = neuron_num.reshape(-1, 1)
    neuron_type = torch.floor(ratio * neuron_num).int() + regions[:, 0].reshape(-1, 1)
    return neuron_type

def syn_within_region(syn_num, region):
    start = 1
    for neurons in region:
        if start:
            syn = torch.randint(neurons[0], neurons[1],
                            size=((neurons[1]-neurons[0]) * syn_num, 2), device=device)
            start = 0
        else:
            syn = torch.concat((syn, torch.randint(neurons[0], neurons[1],
                            size=((neurons[1]-neurons[0]) * syn_num, 2), device=device)))
    return syn

def syn_cross_region(weight_matrix, region):
    start = 1
    for i in range(len(weight_matrix)):
        for j in range(len(weight_matrix)):
            if weight_matrix[i][j] < 10:
                continue
            else:
                pre = torch.randint(region[j][0], region[j][1],
                                    size=(weight_matrix[i][j], 1), device=device)
                post = torch.randint(region[i][0], region[i][1],
                                     size=(weight_matrix[i][j], 1), device=device)
                if start:
                    syn = torch.concat((post, pre), dim=1)
                    start = 0
                else:
                    syn = torch.concat((syn, torch.concat((post, pre), dim=1)))
    return syn

size = 100
neuron_model = 'MultiCompartmentaEIF'

weight_matrix = torch.from_numpy(np.load("IIT_connectivity_matrix.npy")[0:84,0:84])
weight_matrix = weight_matrix.int() * 10
# weight_matrix = np.load('./IIT_connectivity_matrix.npy')
# weight_matrix = torch.from_numpy(weight_matrix)

NR = len(weight_matrix)
data = size * np.ones(NR)
neuron_num = np.array(data).astype(np.int32)
neuron_num = torch.from_numpy(neuron_num)
print(torch.sum(neuron_num))
regions = brain_region(neuron_num)
ratio = torch.tensor([[0.7, 0.9, 1.0] * NR]).reshape(NR, 3)
neuron_types = neuron_type(neuron_num, ratio, regions)
syn_1 = syn_within_region(10, regions)
syn_2 = syn_cross_region(weight_matrix, regions)
syn = torch.concat((syn_1, syn_2))
print(len(syn_2))


print(syn.shape)
weight = -torch.ones(len(syn), device=device, requires_grad=False)
if neuron_model == 'aEIF':
    threshold = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    v_reset = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    c_m = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    tao_w = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    alpha_ad = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    beta_ad = torch.zeros(regions[-1][1], device=device, requires_grad=False)
elif neuron_model == 'HH':
    threshold = torch.zeros(regions[-1][1], device=device, requires_grad=False)

if neuron_model == 'MultiCompartmentaEIF':
    threshold = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    v_reset = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    c_m = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    tao_w = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    alpha_ad = torch.zeros(regions[-1][1], device=device, requires_grad=False)
    beta_ad = torch.zeros(regions[-1][1], device=device, requires_grad=False)
for i in range(len(neuron_types)):
    pre = syn[:, 0]
    mask = (pre >= regions[i][0]) & (pre < neuron_types[i][0])
    indices = torch.where(mask)
    weight[indices] = 1.5
    if neuron_model == 'aEIF':
        if i < 70:
            threshold[regions[i][0]:neuron_types[i][0]] = -50
            threshold[neuron_types[i][0]:neuron_types[i][1]] = -44
            threshold[neuron_types[i][1]:neuron_types[i][2]] = -45
            v_reset[regions[i][0]:neuron_types[i][0]] = -110
            v_reset[neuron_types[i][0]:neuron_types[i][1]] = -110
            v_reset[neuron_types[i][1]:neuron_types[i][2]] = -110
            c_m[regions[i][0]:neuron_types[i][0]] = 10
            c_m[neuron_types[i][0]:neuron_types[i][1]] = 10
            c_m[neuron_types[i][1]:neuron_types[i][2]] = 8.5
            tao_w[regions[i][0]:neuron_types[i][0]] = 1
            tao_w[neuron_types[i][0]:neuron_types[i][1]] = 2
            tao_w[neuron_types[i][1]:neuron_types[i][2]] = 2
            alpha_ad[regions[i][0]:neuron_types[i][0]] = 0
            alpha_ad[neuron_types[i][0]:neuron_types[i][1]] = -0.2
            alpha_ad[neuron_types[i][1]:neuron_types[i][2]] = -0.2
            beta_ad[regions[i][0]:neuron_types[i][0]] = 0
            beta_ad[neuron_types[i][0]:neuron_types[i][1]] = 0.45
            beta_ad[neuron_types[i][1]:neuron_types[i][2]] = 0.45
        else:
            threshold[regions[i][0]:neuron_types[i][0]] = -50
            threshold[neuron_types[i][0]:neuron_types[i][1]] = -50
            threshold[neuron_types[i][1]:neuron_types[i][2]] = -45
            v_reset[regions[i][0]:neuron_types[i][0]] = -100
            v_reset[neuron_types[i][0]:neuron_types[i][1]] = -100
            v_reset[neuron_types[i][1]:neuron_types[i][2]] = -105
            c_m[regions[i][0]:neuron_types[i][0]] = 20
            c_m[neuron_types[i][0]:neuron_types[i][1]] = 10
            c_m[neuron_types[i][1]:neuron_types[i][2]] = 10
            tao_w[regions[i][0]:neuron_types[i][0]] = 1
            tao_w[neuron_types[i][0]:neuron_types[i][1]] = 2
            tao_w[neuron_types[i][1]:neuron_types[i][2]] = 2
            alpha_ad[regions[i][0]:neuron_types[i][0]] = 0
            alpha_ad[neuron_types[i][0]:neuron_types[i][1]] = -0.2
            alpha_ad[neuron_types[i][1]:neuron_types[i][2]] = -0.2
            beta_ad[regions[i][0]:neuron_types[i][0]] = 0
            beta_ad[neuron_types[i][0]:neuron_types[i][1]] = 0.45
            beta_ad[neuron_types[i][1]:neuron_types[i][2]] = 0.45
    elif neuron_model == 'HH':
        threshold[regions[i][0]:neuron_types[i][0]] = 20
        threshold[neuron_types[i][0]:neuron_types[i][1]] = 20
        threshold[neuron_types[i][1]:neuron_types[i][2]] = 20
    
    elif neuron_model == 'MultiCompartmentaEIF':
        if i < 70:
            threshold[regions[i][0]:neuron_types[i][0]] = -50.0
            threshold[neuron_types[i][0]:neuron_types[i][1]] = -44.0
            threshold[neuron_types[i][1]:neuron_types[i][2]] = -45.0
            v_reset[regions[i][0]:neuron_types[i][0]] = -110.0
            v_reset[neuron_types[i][0]:neuron_types[i][1]] = -110.0
            v_reset[neuron_types[i][1]:neuron_types[i][2]] = -110.0
            c_m[regions[i][0]:neuron_types[i][0]] = 10.0
            c_m[neuron_types[i][0]:neuron_types[i][1]] = 10.0
            c_m[neuron_types[i][1]:neuron_types[i][2]] = 8.5
            tao_w[regions[i][0]:neuron_types[i][0]] = 1
            tao_w[neuron_types[i][0]:neuron_types[i][1]] = 2
            tao_w[neuron_types[i][1]:neuron_types[i][2]] = 2
            alpha_ad[regions[i][0]:neuron_types[i][0]] = 0
            alpha_ad[neuron_types[i][0]:neuron_types[i][1]] = -0.2
            alpha_ad[neuron_types[i][1]:neuron_types[i][2]] = -0.2
            beta_ad[regions[i][0]:neuron_types[i][0]] = 0
            beta_ad[neuron_types[i][0]:neuron_types[i][1]] = 0.45
            beta_ad[neuron_types[i][1]:neuron_types[i][2]] = 0.45
        else:
            threshold[regions[i][0]:neuron_types[i][0]] = -50.0
            threshold[neuron_types[i][0]:neuron_types[i][1]] = -50.0
            threshold[neuron_types[i][1]:neuron_types[i][2]] = -45.0
            v_reset[regions[i][0]:neuron_types[i][0]] = -100.0
            v_reset[neuron_types[i][0]:neuron_types[i][1]] = -100.0
            v_reset[neuron_types[i][1]:neuron_types[i][2]] = -105.0
            c_m[regions[i][0]:neuron_types[i][0]] = 20
            c_m[neuron_types[i][0]:neuron_types[i][1]] = 10
            c_m[neuron_types[i][1]:neuron_types[i][2]] = 10
            tao_w[regions[i][0]:neuron_types[i][0]] = 1
            tao_w[neuron_types[i][0]:neuron_types[i][1]] = 2
            tao_w[neuron_types[i][1]:neuron_types[i][2]] = 2
            alpha_ad[regions[i][0]:neuron_types[i][0]] = 0
            alpha_ad[neuron_types[i][0]:neuron_types[i][1]] = -0.2
            alpha_ad[neuron_types[i][1]:neuron_types[i][2]] = -0.2
            beta_ad[regions[i][0]:neuron_types[i][0]] = 0
            beta_ad[neuron_types[i][0]:neuron_types[i][1]] = 0.45
            beta_ad[neuron_types[i][1]:neuron_types[i][2]] = 0.45

if neuron_model == 'aEIF':
    p_neuron = [threshold, v_reset, c_m, tao_w, alpha_ad, beta_ad]
    dt = 1
    T = 2000
elif neuron_model == 'HH':
    p_neuron = [threshold, 120, 36, 0.3, 115, -12, 10.6, 1]
    dt = 0.01
    T = 10000
elif neuron_model == 'MultiCompartmentaEIF':
    p_neuron = [threshold, v_reset, c_m, tao_w, alpha_ad, beta_ad]
    dt = 1.0
    T = 2000
model = brain(syn, weight, neuron_model, p_neuron, dt, device)
# device_ids = [0,2,3,4,5,7,8,9]
# model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)

def neuron_delete(model, rate):
    neuron_idex = torch.arange(0, model.neuron_num)
    delete_num = int(model.neuron_num * rate)
    random_elements = neuron_idex[torch.randperm(model.neuron_num)[:delete_num]]
    model.neurons.threshold[random_elements] = 1000
    return model.neuron_num - delete_num

def syn_delete(model, rate):
    indices = model.syns.w._indices()
    values = model.syns.w._values()
    delete_num = int(len(values) * rate)
    syn_idex = torch.arange(0, len(values))
    random_elements = syn_idex[torch.randperm(len(values))[:delete_num]]
    new_values = values[random_elements]
    new_indices = indices[:, random_elements]
    new_w = torch.sparse_coo_tensor(new_indices, new_values, size=(model.neuron_num, model.neuron_num))
    model.syns.w = new_w

def syn_strength(model, rate):
    indices = model.syns.w._indices()
    values = model.syns.w._values()
    iex = torch.where(values>0)
    values[iex] = values[iex] * rate
    new_values = values
    new_indices = indices
    new_w = torch.sparse_coo_tensor(new_indices, new_values, size=(model.neuron_num, model.neuron_num))
    model.syns.w = new_w

Iraster = []
fire_rate = []
count_n = model.neuron_num
for t in range(T):
    if t == int(T/4):
        count_n = neuron_delete(model, 0.4)
    if t == int(T/4 * 2):
        syn_delete(model, 0.4)
    if t == int(T/4 * 3):
        syn_strength(model, 3)
    model(0)
    # print(torch.sum(model.neurons.spike))
    Isp = torch.nonzero(model.neurons.spike)
    print(len(Isp))
    fire_rate.append(len(Isp)/count_n)
    if (len(Isp) != 0):
        left = t * torch.ones((len(Isp)), device=device, requires_grad=False)
        left = left.reshape(len(left), 1)
        mide = torch.concat((left, Isp), dim=1)
    if (len(Isp) != 0) and (len(Iraster) != 0):
        Iraster = torch.concat((Iraster, mide), dim=0)
    if (len(Iraster) == 0) and (len(Isp) != 0):
        Iraster = mide

torch.save(fire_rate, './fire_rate.pt')
plt.plot(fire_rate)
plt.xlabel('time/mm')
plt.ylabel('fire_rate')
# plt.axvline(x=[500, 1000, 1500], color='b', linestyle='--')
plt.show()
Iraster = torch.tensor(Iraster).transpose(0, 1)
torch.save(Iraster, "./human_MultiCompartmentaEIF100.pt")
Iraster = Iraster.cpu()
plt.figure(figsize=(15, 15))
plt.scatter(Iraster[0], Iraster[1], c='k', marker='.', s=0.001)
plt.savefig('mouse_MultiCompartmentaEIF100.png')
plt.show()
