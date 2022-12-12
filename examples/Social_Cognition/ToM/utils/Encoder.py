import numpy as np
import torch.nn as nn


#exploit or explore
num_enpop = 6
num_depop = 10

class PopEncoder(nn.Module):
    """
    One kind of population coding
    """
    def __init__(self, step, encode_type):
        super(PopEncoder, self).__init__()
        self.step = step
        self.fun = getattr(self, encode_type)
        self.encode_type = encode_type

    def forward(self, inputs, *args, **kwargs):
        outputs = self.fun(inputs, *args, **kwargs)
        return outputs

    def rate(self, inputs, pop , num_state):
        I = np.zeros((pow(num_enpop, num_state), self.step)) #将每一个状态都用一个神经元表示
        #obs /in [1,2,3,4,5] ; obs_py /in [0,1,2,3,4]
        obs_py = []
        for i in range(len(inputs)):
            obs_py.append(inputs[i]-1)
        # six进制
        ind = 0
        for j in range(num_state): #cell_num
            ind += pow(num_enpop, num_state - j - 1) * obs_py[j]

        I[ind, 0: self.step] = 2

        return I