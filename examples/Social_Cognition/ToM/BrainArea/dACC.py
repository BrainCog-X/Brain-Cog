import torch
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
from utils.one_hot import *
import os
import time
import sys
from tqdm import tqdm

from braincog.base.encoder.population_coding import *
from braincog.model_zoo.base_module import BaseLinearModule, BaseModule
from braincog.base.learningrule.STDP import *
import sys
sys.path.append("..")

class dACC(BaseModule):
    """
    SNNLinear
    """
    def __init__(self,
                 step,
                 encode_type,
                 in_features:int,
                 out_features:int,
                 bias,
                 node,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.node1 = node(threshold=0.5, tau=2.)
        self.node_name1 = node
        self.node2 = node(threshold=0.1, tau=2.)
        self.node_name2 = node
        self.fc = self._create_fc()
        self.c = self._rest_c()


    def _rest_c(self):
        c = torch.rand((self.out_features, self.in_features)) # eligibility trace
        return c

    def _create_fc(self):
        """
        the connection of the SNN linear
        @return: nn.Linear
        """
        fc = nn.Linear(in_features=self.in_features,
                  out_features=self.out_features, bias=self.bias)
        return fc

    def update_c(self, c, STDP, tau_c=0.2):
        """
        update the trace of eligibility
        @param c: a tensor to record eligibility
        @param STDP: the results of STDP
        @param tau_c: the parameter of trace decay
        @return: a update tensor to record eligibility
        Equation:
        delta_c = (-(c / tau_c) + STDP) * dela_t
        c = c + delta_c
        reference:<Solving the Distal Reward Problem through ...>
        """
        c = c + tau_c * STDP
        return c

    def forward(self, inputs, epoch):
        """
        decision
        @param inputs: state
        @return: action
        """
        output = []
        stdp = STDP(self.node2, self.fc, decay=0.80)
        self.c = self._rest_c()
        # stdp.connection.weight.data = torch.rand((self.out_features, self.in_features))

        for i in range(inputs.shape[0]):
            for t in range(self.step):
                l1_in = torch.tensor(inputs[i, :])
                l1_out = self.node1(l1_in).unsqueeze(0)  #pre  : l1_out
                l2_out, dw = stdp(l1_out)   #dw -- STDP
                self.c = self.update_c(self.c, dw[0])
            output.append(torch.min(l2_out))
            # output.append((l2_out.any() == 0).cpu().detach().numpy().tolist())

        return output


# if __name__ == '__main__':
#     np.random.seed(6)
#     T = 5
#     num_popneurons = 2
#     safety = 2
#     epoch = 50
#     file_name = "/home/zhaozhuoya/braincog/examples/ToM/data/injury_value.txt"
#     state = []
#     with open(file_name) as f:
#         data = []
#         data_split = f.readlines()  #
#         for i in data_split:
#             state.append(one_hot(int(i[0])))
#
#     output = np.array(state)
#     train_y = output
#     test_y = output[79:82]#output[12].reshape(1,2)
#
#     file_name = "/home/zhaozhuoya/braincog/examples/ToM/data/injury_memory.txt"
#     state = []
#     with open(file_name) as f:
#         data_split = f.readlines()
#         for i in data_split:
#             data = []
#             data.append(int(bool(abs(int(i[2]) - int(i[18]))))*10)
#             data.append(int(bool(abs(int(i[5]) - int(i[21]))))*10)
#             state.append(data)
#     input = np.array(state)
#     train_x = input
#     test_x = input[79:82]
#     dACC_net = dACC(step=T, encode_type='rate', bias=True,
#                         in_features=num_popneurons, out_features=safety,
#                         node=node.LIFNode)
#     dACC_net.fc.weight.data = torch.rand((safety, num_popneurons))
#     dACC_net.load_state_dict(torch.load('./checkpoint/dACC_net.pth')['dacc'])
#     output = dACC_net(inputs=train_x, epoch=50)
#     for i in range(len(output)):
#         print(output[i], train_x[i])
    # torch.save({'dacc': dACC_net.state_dict()}, os.path.join('./checkpoint', 'dACC_net.pth'))
    # dACC_net.load_state_dict(torch.load('./checkpoint/dACC_net.pth')['dacc'])
    # output = dACC_net(inputs=test_x, epoch=50)
    # for i in range(len(test_x)):
    #
    #     print(output[i],test_x[i])


