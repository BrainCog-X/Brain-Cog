import numpy as np
from scipy.linalg import orth
from scipy.special import expit
from scipy.signal import fftconvolve
import torch
from torch.nn import Parameter
import torch.nn as nn
from braincog.datasets.gen_input_signal import img2spikes, lambda_max, dt

gamma = 0.1
beta = 1.0
theta = 3.0

# kernel parameters
tau_s = 4.0  # synaptic time constant
tau_L = 10.0  # leak time constant

# conductance parameters
g_B = 0.6                                   # basal conductance
g_A = 0.05                                  # apical conductance
g_L = 1.0 / tau_L                             # leak conductance
g_D = g_B                                   # dendritic conductance in output layer

k_D = g_D / (g_L + g_D)


STEPS = int(50 / dt)
SLEN = 20                      # spike time length

# --- sigmoid function --- #


def sigma(x):
    return torch.sigmoid(x)

# def sigma(x):
#     return gamma * np.log(1+np.exp(beta*(x-theta)))


def deriv_sigma(x):
    return sigma(x) * (1.0 - sigma(x))


# kernel parameters
tau_s = 4.0                                                   # synaptic time constant
tau_L = 10.0                                                # leak time constant
# --- kernel function --- #
mem = STEPS


def kappa(x):
    return np.exp(-x / tau_s)


def get_kappas(n):
    return np.array([kappa(i + 1) for i in range(n)])


kappas = get_kappas(mem // 2)  # initialize kappas array
kernel = np.zeros(mem)

kernel[:mem // 2] = kappas[:]
kernel[mem // 2:] = -np.flipud(kappas)[:]


W_MIN = -1.0
W_MAX = 1.0

# PID
q1 = 1.0
q2 = 0.1
q3 = 0.001


class Net(nn.Module):
    """
    两房室脉冲神经网络
    """
    def __init__(self, net_size):
        super().__init__()
        self.input_size = net_size[0]
        self.hidden_layers = nn.ModuleList([Hidden_layer(net_size[i], net_size[i + 1], net_size[-1]) for i in range(len(net_size) - 2)])
        self.out_layer = Output_layer(net_size[-2], net_size[-1])
        self.kernel = torch.from_numpy(kernel[:, np.newaxis]).cuda()

    def update_state(self, input_, label, test):
        if len(self.hidden_layers) > 1:
            self.hidden_layers[0].update_state(input_, self.out_layer.spike_rate, test=test)
            for i in range(len(self.hidden_layers) - 2):
                self.hidden_layers[i + 1].update_state(self.hidden_layers[i].spike_rate, self.out_layer.spike_rate, test=test)

            self.hidden_layers[-1].update_state(self.hidden_layers[-2].spike_rate, self.out_layer.spike_rate, test=test)
        else:
            self.hidden_layers[0].update_state(input_, self.out_layer.spike_rate, test=test)

        self.out_layer.update_state(self.hidden_layers[-1].spike_rate, label, test=test)

    def routine(self,
                input_,
                input_delta,
                image_ori,
                image_ori_delta,
                shift,
                label,
                test=False,
                noise=False,
                noise_rate=None):
        """
        网络信息处理过程
        :param input_: 输入图片
        :param input_delta: 输入扰动图片，用于计算相位
        :param image_ori: 原始图片
        :param image_ori_delta: 原始扰动图片
        :param shift: 是否反转背景
        :param label: 输入数据分类标签
        :param test: 是否是测试阶段
        :param noise: 是否增加噪声
        :param noise_rate: 噪声比例
        """
        input_ = img2spikes(input_, input_delta, image_ori, image_ori_delta, STEPS, SLEN, shift, noise=noise, noise_rate=noise_rate)
        input_ = torch.from_numpy(input_).to(self.kernel.device)
        psp = torch.mm(input_, self.kernel).abs().float()

        for i in range(STEPS):
            self.update_state(psp, label, test=test)

    def update_weight(self, lr, t, beta, eps):
        self.out_layer.update_weight(lr, t, beta, eps)
        if len(self.hidden_layers) > 1:
            self.hidden_layers[-1].update_weight(self.out_layer.delta, lr, t, beta, eps)
            for i in range(len(self.hidden_layers) - 1):
                self.hidden_layers[-(i + 2)].update_weight(self.hidden_layers[-(i + 1)].delta, lr, t, beta, eps)
        else:
            self.hidden_layers[0].update_weight(self.out_layer.delta, lr, t, beta, eps)

    def predict(self,
                input_,
                input_delta,
                image_ori,
                image_ori_delta,
                shift,
                noise,
                noise_rate=0):
        self.routine(input_,
                     input_delta,
                     image_ori=image_ori,
                     image_ori_delta=image_ori_delta,
                     shift=shift,
                     label=None,
                     test=True,
                     noise=noise,
                     noise_rate=noise_rate)

        pred = torch.argmax(self.out_layer.spike_rate.flatten())
        return pred


class Hidden_layer(nn.Module):
    """
    隐藏层两房室网络
    """
    def __init__(self, input_size, neu_num, fb_neus):
        super().__init__()
        self.basal_linear = nn.Linear(input_size, neu_num)
        nn.init.uniform_(self.basal_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.basal_linear.bias, -0.1, 0.1)
        self.soma_V = 0.0
        self.basal_V = 0.0
        # for adam
        self.m = 0.0
        self.v = 0.0
        self.m_hat = 0.0
        self.v_hat = 0.0
        self.m_b = 0.0
        self.v_b = 0.0
        self.m_b_hat = 0.0
        self.v_b_hat = 0.0
        # backprop
        self.delta = 0.0

    def update_state(self, basal_input, apical_input, test):
        self.basal_input = basal_input.T  # [1, 781]
        self.basal_V = self.basal_linear(basal_input.T)
        self.soma_V = self.soma_V + 1 / tau_L * (-self.soma_V + g_B / g_L * (self.basal_V - self.soma_V)) * dt
        self.spike_rate = lambda_max * sigma(self.soma_V)

    def update_weight(self, delta_, lr, t, beta, eps):
        weight_dot = lambda_max * k_D * delta_ * deriv_sigma(k_D * self.basal_V)  # [1, 500]
        self.delta = torch.mm(weight_dot, self.basal_linear.weight.data)  # [500, 784] x [1, 500]
        weight_delta = weight_dot[:, :, None] * self.basal_input[:, None, :]
        bias_delta = weight_dot
        self.m = beta[0] * self.m + (1 - beta[0]) * weight_delta
        self.v = beta[1] * self.v + (1 - beta[1]) * torch.square(weight_delta)
        self.m_hat = self.m / (1 - beta[0] ** t)
        self.v_hat = self.v / (1 - beta[1] ** t)
        self.m_b = beta[0] * self.m_b + (1 - beta[0]) * bias_delta
        self.v_b = beta[1] * self.v_b + (1 - beta[1]) * torch.square(bias_delta)
        self.m_b_hat = self.m_b / (1 - beta[0] ** t)
        self.v_b_hat = self.v_b / (1 - beta[1] ** t)
        # update weight
        weight_delta = lr * self.m_hat / (torch.sqrt(self.v_hat) + eps)
        bias_delta = lr * self.m_b_hat / (torch.sqrt(self.v_b_hat) + eps)
        self.basal_linear.weight.data.sub_(weight_delta.mean(0))
        self.basal_linear.bias.data.sub_(bias_delta.mean(0))


class Output_layer(nn.Module):
    """
    输出层两房室网络
    """
    def __init__(self, input_size, neu_num):
        super().__init__()
        self.basal_linear = nn.Linear(input_size, neu_num)
        nn.init.uniform_(self.basal_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.basal_linear.bias, -0.1, 0.1)
        self.soma_V = 0.0
        self.basal_V = 0.0
        self.spike_rate = 0.0
        # adam
        self.m = 0.0
        self.v = 0.0
        self.m_hat = 0.0
        self.v_hat = 0.0
        self.m_b = 0.0
        self.v_b = 0.0
        self.m_b_hat = 0.0
        self.v_b_hat = 0.0
        # backprop
        self.delta = 0.0

    def update_state(self, basal_input, I, test):
        self.basal_input = basal_input
        self.basal_V = self.basal_linear(basal_input)
        if test:
            self.soma_V = self.soma_V + 1 / tau_L * (-self.soma_V + g_B / g_L * (self.basal_V - self.soma_V)) * dt
        else:
            self.soma_V = self.soma_V + 1 / tau_L * (-self.soma_V + g_B / g_L * (self.basal_V - self.soma_V) +
                                                     I - self.soma_V) * dt
        self.spike_rate = lambda_max * sigma(self.soma_V)

    def update_weight(self, lr, t, beta, eps):
        weight_dot = lambda_max * k_D * (sigma(k_D * self.basal_V) - sigma(self.soma_V)) * deriv_sigma(k_D * self.basal_V)
        self.delta = torch.mm(weight_dot, self.basal_linear.weight.data)     # [1, 500]
        bias_delta = weight_dot  # [1, 10]
        weight_delta = weight_dot[:, :, None] * self.basal_input[:, None, :]   # [1, 10, 500]
        self.m = beta[0] * self.m + (1 - beta[0]) * weight_delta
        self.v = beta[1] * self.v + (1 - beta[1]) * torch.square(weight_delta)
        self.m_hat = self.m / (1 - beta[0] ** t)
        self.v_hat = self.v / (1 - beta[1] ** t)
        self.m_b = beta[0] * self.m_b + (1 - beta[0]) * bias_delta
        self.v_b = beta[1] * self.v_b + (1 - beta[1]) * torch.square(bias_delta)
        self.m_b_hat = self.m_b / (1 - beta[0] ** t)
        self.v_b_hat = self.v_b / (1 - beta[1] ** t)
        # update weight
        weight_delta = lr * self.m_hat / (torch.sqrt(self.v_hat) + eps)
        bias_delta = lr * self.m_b_hat / (torch.sqrt(self.v_b_hat) + eps)
        self.basal_linear.weight.data.sub_(weight_delta.mean(0))
        self.basal_linear.bias.data.sub_(bias_delta.mean(0))
