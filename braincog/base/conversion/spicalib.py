import torch
import torch.nn as nn


class SpiCalib(nn.Module):
    def __init__(self, allowance):
        super(SpiCalib, self).__init__()
        self.allowance = allowance
        self.sumspike = 0
        self.t = 0

    def forward(self, x):
        if self.allowance == 0:
            return x

        if self.t == 0:
            self.last_spike = torch.zeros_like(x)
            self.avg_time = torch.zeros_like(x)
            self.num_spike = torch.zeros_like(x)

        SPIKE_MASK = x > 0
        self.num_spike[SPIKE_MASK] += 1
        self.avg_time[SPIKE_MASK] = (self.t - self.last_spike + self.avg_time * (self.num_spike - 1))[SPIKE_MASK] / \
                                    self.num_spike[SPIKE_MASK]
        self.last_spike[SPIKE_MASK] = self.t
        SIN_MASK = self.t - self.last_spike > self.avg_time + self.allowance
        x[SIN_MASK] -= 1.0
        self.sumspike += x
        x[self.sumspike <= -1] = 0
        self.t += 1
        return x

    def reset(self):
        self.sumspike = 0
        self.t = 0