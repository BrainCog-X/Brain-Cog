from signal import signal
from subprocess import call
import numpy as np
import random
import copy


class QSEncoder:
    """
    QS Encoding.
    :param lambda_max: 最大发放率
    :param steps: 脉冲发放周期长度 T
    :param sig_len: 脉冲发放窗口
    :param shift: 是否反转背景
    :param noise: 是否增加噪声
    :param noise_rate: 噪声比例
    :param eps: 防止溢出参数
    """
    def __init__(self,
        lambda_max,
        steps,
        sig_len,
        shift=False,
        noise=None,
        noise_rate=None,
        eps=1e-6
    ) -> None:
        self._lambda_max = lambda_max
        self._steps = steps
        self._sig_len = sig_len
        self._shift = shift
        self._noise = noise
        self._noise_rate = noise_rate
        self._eps = eps


    def __call__(self, image, image_delta, image_ori, image_ori_delta):
        """
        将图片转换为脉冲。
        :param image: 背景反转图片
        :param image_delta: 扰动图片，用于计算相位
        :param image_ori: 原始图片
        :param image_ori_delta: 原始扰动图片
        """
        if self._noise:
            signals = self.noise_trans(image, image_ori, image_ori_delta)
        elif self._shift:
            signals = self.shift_trans(image, image_delta, image_ori, image_ori_delta)
        else:
            signals = np.zeros((self.steps, image.shape[0]))
            signal_possion = np.random.poisson(image, (self._sig_len, image.shape[0]))
            signals[:self._sig_len] = signal_possion[:]
        return signal.T


    def shift_trans(self, image, image_delta, image_ori, image_ori_delta):
        """
        背景翻转图片转脉冲序列。
        :param image: 背景反转图片
        :param image_delta: 扰动图片，用于计算相位
        :param image_ori: 原始图片
        :param image_ori_delta: 原始扰动图片
        """
        signal = np.zeros((self._steps, image.shape[0]))
        assert image_ori is not None
        assert self.noise is False
        assert image_delta is not None
        assert image_ori_delta is not None
        image_ori_reverse = self._lambda_max - image_ori
        image_ori_delta_reverse = self._lambda_max - image_ori_delta
        zeta = image / (image_ori**2 + image_ori_reverse**2) ** 0.5
        zeta_delta = image_delta / (image_ori_delta**2 + image_ori_delta_reverse**2)**0.5
        idx_left = zeta < zeta_delta
        phi = np.arctan(image_ori / (image_ori_reverse + self._eps))
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
        spike_rate = np.abs((self._lambda_max * sin_theta - image) / (sin_theta - cos_theta + self._eps))
        signal_possion = np.random.poisson(spike_rate, (self._sig_len, spike_rate.shape[0]))
        shift_step = np.rint(np.clip(2 * theta / np.pi, a_min=0, a_max=1.0) * (self._steps - self._sig_len))
        shift_step = shift_step.astype(np.int)
        signal[shift_step:shift_step + self._sig_len] = signal_possion[:]



    def noise_trans(self, image, image_ori, image_ori_delta):
        """
        噪声图片转脉冲序列
        :param image: 背景反转图片
        :param image_ori: 原始图片
        :param image_ori_delta: 原始扰动图片
        """
        signal = np.zeros((self._steps, image.shape[0]))
        assert image_ori is not None
        assert self._shift is False
        assert self._noise_rate is not None
        image_ori_delta = copy.deepcopy(image_ori)
        idx = image_ori_delta < (self._lambda_max - 0.001)
        image_ori_delta[idx] += 0.001
        image_ori_reverse = self._lambda_max - image_ori
        image_ori_delta_reverse = self._lambda_max - image_ori_delta
        image_noise, image_delta_noise = self.reverse_pixels(image_ori, image_ori_delta, noise_rate=self._noise_rate)
        zeta = image_noise / (image_ori**2 + image_ori_reverse**2)**0.5
        zeta_delta = image_delta_noise / (image_ori_delta**2 + image_ori_delta_reverse**2)**0.5
        idx_left = zeta < zeta_delta
        phi = np.arctan(image_ori / (image_ori_reverse + self._eps))
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
        spike_rate = np.abs((self._lambda_max * sin_theta - image_noise) / (sin_theta - cos_theta + self._eps))
        signal_possion = np.random.poisson(spike_rate, (self._sig_len, spike_rate.shape[0]))
        shift_step = np.rint(np.clip(2 * theta / np.pi, a_min=0, a_max=1.0) * (self._steps - self._sig_len))
        shift_step = shift_step.astype(np.int)
        signal[shift_step:shift_step + self._sig_len] = signal_possion[:]
        return signal

    def reverse_pixels(self, image, image_delta, noise_rate, flip_bits=None):
        """
        反转图片像素
        """
        if flip_bits is None:
            N = int(noise_rate * image.shape[0])
            flip_bits = random.sample(range(image.shape[0]), N)
            img = copy.copy(image)
            img_delta = copy.copy(image_delta)

            img[flip_bits] = self._lambda_max - img[flip_bits]
            img_delta[flip_bits] = self._lambda_max - img_delta[flip_bits]
        return img, img_delta