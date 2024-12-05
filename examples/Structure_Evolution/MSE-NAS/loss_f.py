import torch
import torch.nn.functional as f


def psp(inputs, n_steps,tau_s):
    shape = inputs.shape
    n_steps = n_steps
    tau_s = tau_s

    syn = torch.zeros(shape[0], shape[1], shape[2], shape[3]).cuda()
    syns = torch.zeros(shape[0], shape[1], shape[2], shape[3], n_steps).cuda()

    for t in range(n_steps):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s

    return syns


class SpikeLoss(torch.nn.Module):
    """
    This class defines different spike based loss modules that can be used to optimize the SNN.
    """
    def __init__(self, desired_count,undesired_count):
        super(SpikeLoss, self).__init__()
        self.desired_count = desired_count
        self.desired_count = undesired_count
        self.criterion = torch.nn.CrossEntropyLoss()

    def spike_count(self, outputs, target, desired_count,undesired_count):
        delta = loss_count.apply(outputs, target, desired_count,undesired_count)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_kernel(self, outputs, target, desired_count,undesired_count):
        delta = loss_kernel.apply(outputs, target, desired_count,undesired_count)
        return 1 / 2 * torch.sum(delta ** 2)

    def spike_soft_max(self, outputs, target):
        delta = f.log_softmax(outputs.sum(dim=4).squeeze(-1).squeeze(-1), dim = 1)
        return self.criterion(delta, target)


class loss_count(torch.autograd.Function):  # a and u is the incremnet of each time steps
    @staticmethod
    def forward(ctx, outputs, target, desired_count,undesired_count):
        desired_count = desired_count
        undesired_count = undesired_count
        shape = outputs.shape
        n_steps = shape[4]
        out_count = torch.sum(outputs, dim=4)

        delta = (out_count - target) / n_steps
        mask = torch.ones_like(out_count)
        mask[target == undesired_count] = 0
        mask[delta < 0] = 0
        delta[mask == 1] = 0
        mask = torch.ones_like(out_count)
        mask[target == desired_count] = 0
        mask[delta > 0] = 0
        delta[mask == 1] = 0
        delta = delta.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
        return delta

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


class loss_kernel(torch.autograd.Function):  # a and u is the incremnet of each time steps
    @staticmethod
    def forward(ctx, outputs, target, n_steps,tau_s):
        # out_psp = psp(outputs, network_config)
        target_psp = psp(target, n_steps,tau_s)
        delta = outputs - target_psp
        return delta

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None
