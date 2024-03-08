import torch.nn as nn

from braincog.base.node import LIFNode
from braincog.base.node.node import LIFNode, DoubleSidePLIFNode, PLIFNode
from braincog.base.strategy.surrogate import AtanGrad
import torch


class AtanLIFNode(LIFNode):
    def __init__(self, tau=0.5, *args, **kwargs):
        super().__init__(tau, *args, **kwargs)
        self.act_fun = AtanGrad(alpha=1., requires_grad=True)


class BCNoSpikingLIFNode(LIFNode):
    def __init__(self, tau, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def forward(self, dv: torch.Tensor):
        # print("dv: ", dv)
        # print("dv.shape: ", dv.shape)
        self.integral(dv)
        return self.mem


def build_model_snn(in_dim, out_dim, layers, hidden, th=0.5, re=0.0, tau=0.5, activation='LIFNode',
                    normalize=lambda x: x):
    # print("build model snn!")
    # 0.activation换成LIFNode...
    if activation == 'LIFNode':
        node = LIFNode(threshold=th, tau=tau)
    elif activation == 'AtanLIFNode':
        node = AtanLIFNode(tau=tau)
    elif activation == 'BCNoSpikingLIFNode':
        node = BCNoSpikingLIFNode(tau=tau)
    elif activation == 'DoubleSidePLIFNode':
        node = DoubleSidePLIFNode(tau=tau)
    elif activation == 'PLIFNode':
        node = PLIFNode(threshold=th)
    elif activation == 'nn.ELU':
        node = nn.ELU()
    elif activation == 'nn.ReLU':
        node = nn.ReLU()
    elif activation == 'nn.Tanh':
        node = nn.Tanh()
    # 1.是否norm no norm
    model = [normalize(nn.Linear(in_dim, hidden))]
    model += [node]
    for i in range(layers - 1):
        model += [normalize(nn.Linear(hidden, hidden))]
        model += [node]
    model += [normalize(nn.Linear(hidden, out_dim))]
    # 使用第二个归一化,node激活之后还要linear，最后的输出应该还得有个node，将out node定义到外面比较合适
    return nn.Sequential(*model)

def build_model(in_dim, out_dim, layers, hidden, activation, normalize=lambda x: x):
    model = [normalize(nn.Linear(in_dim, hidden))]
    model += [activation()]
    for i in range(layers - 1):
        model += [normalize(nn.Linear(hidden, hidden))]
        model += [activation()]
    model += [normalize(nn.Linear(hidden, out_dim))]
    return nn.Sequential(*model)