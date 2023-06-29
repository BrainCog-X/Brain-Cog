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

device = 'cuda:0'

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
        self.neuron_num = len(p_neuron[0])
        self.syns = Syn(syn, weight, self.neuron_num, 3, 6, dt, device)

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
            syn = torch.concatenate((syn, torch.randint(neurons[0], neurons[1],
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
                    syn = torch.concatenate((post, pre), dim=1)
                    start = 0
                else:
                    syn = torch.concatenate((syn, torch.concatenate((post, pre), dim=1)))
    return syn

size = 500
neuron_model = 'HH'
weight_matrix = torch.tensor(genfromtxt("./human.csv", delimiter=',', skip_header=False)) * 100
weight_matrix = weight_matrix.int()

NR = len(weight_matrix)
data = size * np.ones(NR)
neuron_num = np.array(data).astype(np.int32)
neuron_num = torch.from_numpy(neuron_num)
regions = brain_region(neuron_num)
ratio = torch.tensor([[0.7, 0.9, 1.0] * NR]).reshape(NR, 3)
neuron_types = neuron_type(neuron_num, ratio, regions)
syn_1 = syn_within_region(10, regions)
syn_2 = syn_cross_region(weight_matrix, regions)
syn = torch.concatenate((syn_1, syn_2))
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
for i in range(len(neuron_types)):
    pre = syn[:, 0]
    mask = (pre >= regions[i][0]) & (pre < neuron_types[i][0])
    indices = torch.where(mask)
    weight[indices] = 1.5
    if neuron_model == 'aEIF':
        if i < 177:
            threshold[regions[i][0]:neuron_types[i][0]] = -50
            threshold[neuron_types[i][0]:neuron_types[i][1]] = -44
            threshold[neuron_types[i][1]:neuron_types[i][2]] = -45
            v_reset[regions[i][0]:neuron_types[i][0]] = -110
            v_reset[neuron_types[i][0]:neuron_types[i][1]] = -110
            v_reset[neuron_types[i][1]:neuron_types[i][2]] = -66
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
            v_reset[regions[i][0]:neuron_types[i][0]] = -60
            v_reset[neuron_types[i][0]:neuron_types[i][1]] = -60
            v_reset[neuron_types[i][1]:neuron_types[i][2]] = -65
            c_m[regions[i][0]:neuron_types[i][0]] = 20
            c_m[neuron_types[i][0]:neuron_types[i][1]] = 2
            c_m[neuron_types[i][1]:neuron_types[i][2]] = 4
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
        threshold[regions[i][0]:neuron_types[i][0]] = 50
        threshold[neuron_types[i][0]:neuron_types[i][1]] = 60
        threshold[neuron_types[i][1]:neuron_types[i][2]] = 60

if neuron_model == 'aEIF':
    p_neuron = [threshold, v_reset, c_m, tao_w, alpha_ad, beta_ad]
    dt = 1
    T = 300
elif neuron_model == 'HH':
    p_neuron = [threshold, 120, 36, 0.3, 115, -12, 10.6, 1]
    dt = 0.01
    T = 10000
model = brain(syn, weight, neuron_model, p_neuron, dt, device)
Iraster = []
for t in range(T):
    model(0)
    print(torch.sum(model.neurons.spike))
    Isp = torch.nonzero(model.neurons.spike)
    print(len(Isp))
    if (len(Isp) != 0):
        left = t * torch.ones((len(Isp)), device=device, requires_grad=False)
        left = left.reshape(len(left), 1)
        mide = torch.concatenate((left, Isp), dim=1)
    if (len(Isp) != 0) and (len(Iraster) != 0):
        Iraster = torch.concatenate((Iraster, mide), dim=0)
    if (len(Iraster) == 0) and (len(Isp) != 0):
        Iraster = mide

Iraster = torch.tensor(Iraster).transpose(0, 1)
torch.save(Iraster, "./human.pt")
