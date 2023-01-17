import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from braincog.base.node.node import LIFNode

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)    #train
            # self.in_fn = input_dim  #test
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class BCNoSpikingLIFNode(LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dv: torch.Tensor):
        self.integral(dv)
        return self.mem

class SNNNetwork(nn.Module):
    """
    SNN network (can be used as value or policy or MLE)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, node=LIFNode, time_window=16,
                 norm_in=True, output_style='sum'):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(SNNNetwork, self).__init__()

        self._threshold = 0.5
        self.v_reset = 0.0
        self._time_window = time_window
        self.output_style = output_style
        self._node1 = node(threshold=self._threshold, v_reset=self.v_reset)
        self._node2 = node(threshold=self._threshold, v_reset=self.v_reset)

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)    #train
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        if self.output_style == 'sum':
            self._out_node = lambda x: x
        elif self.output_style == 'voltage':
            self._out_node = BCNoSpikingLIFNode()

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

    def forward(self, X):
        qs = []
        self.reset()
        for t in range(self._time_window):
            x = self.fc1((self.in_fn(X)+0.5)) #train
            # x = self.fc1((X + 0.5)) #test
            x = self._node1(x)
            x = self.fc2(x)
            x = self._node2(x)
            x = self.fc3(x)
            x = self._out_node(x)
            qs.append(x)

        if self.output_style == 'sum':
            outputs = sum(qs) / self._time_window
            return outputs
        elif self.output_style == 'voltage':
            outputs = x
            return outputs

