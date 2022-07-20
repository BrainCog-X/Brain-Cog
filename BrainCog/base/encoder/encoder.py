import torch
import torch.nn as nn
from einops import rearrange, repeat
from braincog.base.strategy.surrogate import GateGrad


class Encoder(nn.Module):
    '''
    将static image编码
    :param step: 仿真步长
    :param encode_type: 编码方式, 可选 ``direct``, ``ttfs``, ``rate``, ``phase``
    :param temporal_flatten: 直接将temporal维度concat到channel维度
    :param layer_by_layer: 是否使用计算每一层的所有的输出的方式进行推理
    :param
    (step, batch_size, )
    '''

    def __init__(self, step, encode_type='ttfs', *args, **kwargs):
        super(Encoder, self).__init__()
        self.step = step
        self.fun = getattr(self, encode_type)
        self.encode_type = encode_type
        self.temporal_flatten = kwargs['temporal_flatten'] if 'temporal_flatten' in kwargs else False
        self.layer_by_layer = kwargs['layer_by_layer'] if 'layer_by_layer' in kwargs else False
        # if encode_type == 'auto':
        #     self.fun = AutoEncoder(self.step, spike_output=False)

    def forward(self, inputs, deletion_prob=None, shift_var=None):
        if len(inputs.shape) == 5:  # DVS data
            outputs = inputs.permute(1, 0, 2, 3, 4).contiguous()  # t, b, c, w, h

        else:
            outputs = self.fun(inputs)

        if deletion_prob:
            outputs = self.delete(outputs, deletion_prob)
        if shift_var:
            outputs = self.shift(outputs, shift_var)

        if self.layer_by_layer:
            outputs = rearrange(outputs, 't b c w h -> (t b) c w h')

        return outputs

    @torch.no_grad()
    def direct(self, inputs):
        """
        直接编码
        :param inputs: 形状(b, c, w, h)
        :return: (t, b, c, w, h)
        """
        outputs = repeat(inputs, 'b c w h -> t b c w h', t=self.step)
        return outputs

    @torch.no_grad()
    def ttfs(self, inputs):
        """
        Time-to-First-Spike Encoder
        :param inputs: static data
        :return: Encoded data
        """
        # print("ttfs")
        shape = (self.step,) + inputs.shape
        outputs = torch.zeros(shape, device=self.device)
        for i in range(self.step):
            mask = (inputs * self.step <= (self.step - i)
                    ) & (inputs * self.step > (self.step - i - 1))
            outputs[i, mask] = 1 / (i + 1)
        return outputs

    @torch.no_grad()
    def rate(self, inputs):
        """
        Rate Coding
        :param inputs:
        :return:
        """
        shape = (self.step,) + inputs.shape
        return (inputs > torch.rand(shape, device=self.device)).float()

    @torch.no_grad()
    def phase(self, inputs):
        """
        Phase Coding
        相位编码
        :param inputs: static data
        :return: encoded data
        """
        shape = (self.step,) + inputs.shape
        outputs = torch.zeros(shape, device=self.device)
        inputs = (inputs * 256).long()
        val = 1.
        for i in range(self.step):
            if i < 8:
                mask = (inputs >> (8 - i - 1)) & 1 != 0
                outputs[i, mask] = val
                val /= 2.
            else:
                outputs[i] = outputs[i % 8]
        return outputs

    @torch.no_grad()
    def delete(self, inputs, prob):
        """
        在Coding 过程中随机删除脉冲
        :param inputs: encoded data
        :param prob: 删除脉冲的概率
        :return: 随机删除脉冲之后的数据
        """
        mask = (inputs >= 0) & (torch.randn_like(
            inputs, device=self.device) < prob)
        inputs[mask] = 0.
        return inputs

    @torch.no_grad()
    def shift(self, inputs, var):
        """
        对数据进行随机平移, 添加噪声
        :param inputs: encoded data
        :param var: 随机平移的方差
        :return: shifted data
        """
        outputs = torch.zeros_like(inputs)
        for step in range(self.step):
            shift = (var * torch.randn(1)).round_() + step
            shift.clamp_(min=0, max=self.step - 1)
            outputs[step] += inputs[int(shift)]
        return outputs
