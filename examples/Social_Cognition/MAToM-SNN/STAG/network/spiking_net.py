import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions import Normal

from braincog.base.node.node import IFNode, LIFNode
from braincog.base.strategy.surrogate import AtanGrad


thresh = 0.3
lens = 0.25
decay = 0.3
TIMESTEPS = 15
M = 5


# BrainCog
class BCNoSpikingLIFNode(LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dv: torch.Tensor):
        self.integral(dv)
        return self.mem


class BCNoSpikingIFNode(IFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dv: torch.Tensor):
        self.integral(dv)
        return self.mem


# Sug
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens)


#act_fun = ActFun.apply
act_fun = AtanGrad(alpha=2.,requires_grad=False)

def mem_update(fc, x, mem, spike):
    mem = mem * decay * (1 - spike) + fc(x)
    #spike = act_fun(mem)
    spike = act_fun(x=mem-1)
    return mem, spike


class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.ppo_hidden_size)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias = True)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias = True)
        self.fc4 = nn.Linear(args.rnn_hidden_dim, args.n_actions)#
        self.req_grad = False

    def forward(self, inputs, h1_mem, h1_spike, h2_mem, h2_spike):
        # if self.req_grad == False:
        # [1, 17] -> [1, process, 64]
        x = self.fc1(inputs)
        # x = IFNode()(x)
        x = LIFNode()(x)
        if self.args.alg == 'siql_e':
            h1_mem = h1_mem.reshape(-1,  M,TIMESTEPS, self.args.rnn_hidden_dim)
            h1_spike = h1_spike.reshape(-1,  M,TIMESTEPS, self.args.rnn_hidden_dim)
            h2_mem = h2_mem.reshape(-1, M, TIMESTEPS, self.args.rnn_hidden_dim)
            h2_spike = h2_spike.reshape(-1, M, TIMESTEPS, self.args.rnn_hidden_dim)

        else:
        # [1, 64] -> [process, 64]
            h1_mem = h1_mem.reshape(-1, self.args.rnn_hidden_dim)
            h1_spike = h1_spike.reshape(-1, self.args.rnn_hidden_dim)
            h2_mem = h2_mem.reshape(-1, self.args.rnn_hidden_dim)
            h2_spike = h2_spike.reshape(-1, self.args.rnn_hidden_dim)

        h1_mem, h1_spike = mem_update(self.fc2, x, h1_mem, h1_spike)
        h2_mem, h2_spike = mem_update(self.fc3, h1_spike, h2_mem, h2_spike)
        # [1, 5]
        value = BCNoSpikingLIFNode(tau=2.0)(self.fc4(h2_mem))

        return value, h1_mem, h1_spike, h2_mem, h2_spike




class VDNNet(nn.Module):
    def __init__(self):
        super(VDNNet, self).__init__()

    def forward(self, q_values):
        return torch.sum(q_values, dim=2, keepdim=True)

class Linear_weight(nn.Module):
    def __init__(self, input_shape, out_shape, args):
        super(Linear_weight,self).__init__()
        self.args = args
        # self.fc  = nn.Linear(input_shape, out_shape)
        self.alpha = nn.Parameter(torch.Tensor(out_shape))

    def forward(self, x):
        # return self.fc(x)
        if self.args.alg == 'scovdn_weight':
            x = x[:,:,:,0] * self.alpha + x[:,:,:,1] * (1 - self.alpha)
            return x.unsqueeze(3)
        elif self.args.alg == 'stomvdn':
            x = x[:, :, :, :, 0] * self.alpha + x[:, :, :, :, 1] * (1 - self.alpha)
            return x.unsqueeze(4)

class BiasNet(nn.Module):
    def __init__(self, args):
        super(BiasNet, self).__init__()
        self.args = args
        input_shape = self.args.obs_shape + self.args.rnn_hidden_dim
        #
        # self.h1_mem = self.h1_spike = torch.zeros(self.args.n_episodes * self.args.process,
        #            self.args.episode_limit, self.args.rnn_hidden_dim)
        # if self.args.cuda:
        #     self.h1_mem = self.h1_mem.cuda(self.args.device)
        #     self.h1_spike = self.h1_spike.cuda(self.args.device)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)#neuron.IFNode()
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim, bias = True)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)#

    def reset(self, episode_num):
        self.h1_mem = self.h1_spike = torch.zeros(episode_num,
                   self.args.episode_limit, self.args.rnn_hidden_dim)
        if self.args.cuda:
            self.h1_mem = self.h1_mem.cuda(self.args.device)
            self.h1_spike = self.h1_spike.cuda(self.args.device)

    def forward(self, state, hidden):
        episode_num, max_episode_len, n_agents, _ = hidden.shape
        state = state.reshape(episode_num * max_episode_len, -1)
        state = state * 0.2
        hidden = \
            hidden.reshape(episode_num * max_episode_len, n_agents, -1).sum(dim=-2)
        inputs = torch.cat([state, hidden], dim=-1)

        x = self.fc1(inputs)
        x = neuron.IFNode()(x)
        # x = IFNode()(x)
        # x = LIFNode()(x)      #bad

        # [1, 64] -> [process, 64]
        self.h1_mem = self.h1_mem.reshape(-1, self.args.rnn_hidden_dim)
        self.h1_spike = self.h1_spike.reshape(-1, self.args.rnn_hidden_dim)

        self.h1_mem, self.h1_spike = mem_update(self.fc2, x, self.h1_mem, self.h1_spike)
        # [1, 5]
        # value = NonSpikingLIFNode(tau=2.0)(self.fc4(h2_mem))
        # value = BCNoSpikingLIFNode(tau=2.0)(self.fc4(h2_mem))
        value = BCNoSpikingIFNode(tau=2.0)(self.fc3(self.h1_mem))

        return value
