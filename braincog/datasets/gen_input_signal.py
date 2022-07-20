import numpy as np
import random
import copy

dt = 1.0                  # ms
lambda_max = 0.25 * dt  # maximum spike rate (spikes per time step)
eps_ = 1e-6


def img2spikes(image,
               image_delta,
               image_ori,
               image_ori_delta,
               steps,
               sig_len,
               shift=None,
               noise=None,
               noise_rate=None):
    """
    将图片转换为脉冲。
    :param image: 背景反转图片
    :param image_delta: 扰动图片，用于计算相位
    :param image_ori: 原始图片
    :param image_ori_delta: 原始扰动图片
    :param steps: 脉冲发放周期长度 T
    :param sig_len: 脉冲发放窗口
    :param shift: 是否反转背景
    :param noise: 是否增加噪声
    :param noise_rate: 噪声比例
    """
    signal = np.zeros((steps, image.shape[0]))
    if noise:
        assert image_ori is not None
        assert shift is False
        assert noise_rate is not None
        image_ori_delta = copy.deepcopy(image_ori)
        idx = image_ori_delta < (lambda_max - 0.001)
        image_ori_delta[idx] += 0.001
        image_ori_reverse = lambda_max - image_ori
        image_ori_delta_reverse = lambda_max - image_ori_delta
        image_noise, image_delta_noise = reverse_pixels(image_ori, image_ori_delta, noise_rate=noise_rate)
        zeta = image_noise / (image_ori**2 + image_ori_reverse**2)**0.5
        zeta_delta = image_delta_noise / (image_ori_delta**2 + image_ori_delta_reverse**2)**0.5
        idx_left = zeta < zeta_delta
        phi = np.arctan(image_ori / (image_ori_reverse + eps_))
        zeta = np.clip(zeta, -1, 1)
        zeta = np.arcsin(zeta)
        theta1 = zeta - phi
        theta2 = np.pi - zeta - phi
        theta = np.zeros(theta1.shape)
        theta[idx_left] = theta1[idx_left]
        theta[~idx_left] = theta2[~idx_left]
        theta = np.mean(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        spike_rate = np.abs((lambda_max * sin_theta - image_noise) / (sin_theta - cos_theta + eps_))
        signal_possion = np.random.poisson(spike_rate, (sig_len, spike_rate.shape[0]))
        shift_step = np.rint(np.clip(2 * theta / np.pi, a_min=0, a_max=1.0) * (steps - sig_len))
        shift_step = shift_step.astype(np.int)
        signal[shift_step:shift_step + sig_len] = signal_possion[:]

    elif shift:
        assert image_ori is not None
        assert noise is False
        assert image_delta is not None
        assert image_ori_delta is not None
        image_ori_reverse = lambda_max - image_ori
        image_ori_delta_reverse = lambda_max - image_ori_delta
        zeta = image / (image_ori**2 + image_ori_reverse**2) ** 0.5
        zeta_delta = image_delta / (image_ori_delta**2 + image_ori_delta_reverse**2)**0.5
        idx_left = zeta < zeta_delta
        phi = np.arctan(image_ori / (image_ori_reverse + eps_))
        zeta = np.clip(zeta, -1, 1)
        zeta = np.arcsin(zeta)
        theta1 = zeta - phi
        theta2 = np.pi - zeta - phi
        theta = np.zeros(theta1.shape)
        theta[idx_left] = theta1[idx_left]
        theta[~idx_left] = theta2[~idx_left]
        theta = np.mean(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        spike_rate = np.abs((lambda_max * sin_theta - image) / (sin_theta - cos_theta + eps_))
        signal_possion = np.random.poisson(spike_rate, (sig_len, spike_rate.shape[0]))
        shift_step = np.rint(np.clip(2 * theta / np.pi, a_min=0, a_max=1.0) * (steps - sig_len))
        shift_step = shift_step.astype(np.int)
        signal[shift_step:shift_step + sig_len] = signal_possion[:]

    else:
        signal_possion = np.random.poisson(image, (sig_len, image.shape[0]))
        signal[:sig_len] = signal_possion[:]
    return signal.T


def reverse_pixels(image, image_delta, noise_rate, flip_bits=None):
    """
    反转图片像素
    """
    if flip_bits is None:
        N = int(noise_rate * image.shape[0])
        flip_bits = random.sample(range(image.shape[0]), N)
        img = copy.copy(image)
        img_delta = copy.copy(image_delta)

        img[flip_bits] = lambda_max - img[flip_bits]
        img_delta[flip_bits] = lambda_max - img_delta[flip_bits]
    return img, img_delta


