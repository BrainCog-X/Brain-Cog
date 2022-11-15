import torch
import torch.nn as nn
from einops import rearrange, repeat
from braincog.base.strategy.surrogate import GateGrad


class AutoEncoder(nn.Module):
    def __init__(self, step, spike_output=True):
        super(AutoEncoder, self).__init__()
        self.step = step
        self.spike_output = spike_output

        # self.gru = nn.GRU(input_size=1, hidden_size=1, num_layers=3)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(1, self.step)
        self.fc2 = nn.Linear(self.step, self.step)
        self.relu = nn.ReLU()
        #
        self.act_fun = GateGrad()

    def forward(self, x):
        shape = x.shape

        x = self.fc1(x.view(-1, 1))
        x = self.relu(x)
        x = self.fc2(x).transpose_(1, 0)

        # x = x.view(1, -1, 1).repeat(self.step, 1, 1)
        # x, _ = self.gru(x)

        x = self.sigmoid(x)
        if not self.spike_output:
            return x.view(self.step, *shape)
        else:
            return self.act_fun(x).view(self.step, *shape)


# class TransEncoder(nn.Module):
#     def __init__(self, step):
#         super(TransEncoder, self).__init__()
#         self.step = step
#         self.trans = Transformer(dim=128, depth=3, heads=8, dim_head=, mlp_dim, dropout=0.)


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
        self.no_encode = kwargs['adaptive_node'] if 'adaptive_node' in kwargs else False
        self.groups = kwargs['n_groups'] if 'n_groups' in kwargs else 1
        # if encode_type == 'auto':
        #     self.fun = AutoEncoder(self.step, spike_output=False)

    def forward(self, inputs, deletion_prob=None, shift_var=None):
        if len(inputs.shape) == 5:  # DVS data
            outputs = inputs.permute(1, 0, 2, 3, 4).contiguous()  # t, b, c, w, h

        else:
            if self.encode_type == 'auto':
                if self.fun.device != inputs.device:
                    self.fun.to(inputs.device)
            outputs = self.fun(inputs)

        if deletion_prob:
            outputs = self.delete(outputs, deletion_prob)
        if shift_var:
            outputs = self.shift(outputs, shift_var)

        if self.temporal_flatten or self.no_encode:
            outputs = rearrange(outputs, 't b c w h -> 1 b (t c) w h')
        elif self.groups != 1:
            outputs = rearrange(outputs, 't b c w h -> b (c t) w h')
        elif self.layer_by_layer:
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
        # outputs = inputs.unsqueeze(0).repeat(self.step, *([1] * len(shape)))
        return outputs

    def auto(self, inputs):
        # TODO: Calc loss for firing-rate
        shape = inputs.shape
        outputs = self.fun(inputs)
        print(outputs.shape)
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
        return (inputs > torch.rand(shape, device=inputs.device)).float()

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
        # TODO: Real-time shift
        outputs = torch.zeros_like(inputs)
        for step in range(self.step):
            shift = (var * torch.randn(1)).round_() + step
            shift.clamp_(min=0, max=self.step - 1)
            outputs[step] += inputs[int(shift)]
        return outputs
