import torch
from BrainCog.base.connection.CustomLinear import *
from BrainCog.base.node.node import *
from BrainCog.base.learningrule.STDP import *
from BrainCog.base.brainarea.IPL import *
from BrainCog.base.brainarea.Insula import *

if __name__ == "__main__":
    num_neuron = 4
    num_vPMC = num_neuron
    num_STS  = num_neuron
    num_IPLM = num_neuron
    num_IPLV = num_neuron
    num_Insula = num_neuron

    # InsulaNet
    # connection
    Insula_connection = []
    # IPLV-Insula
    con_matrix0 = torch.eye(num_IPLM, dtype=torch.float) * 2
    Insula_connection.append(CustomLinear(con_matrix0))
    # STS-Insula
    con_matrix1 = torch.eye(num_IPLV, dtype=torch.float) * 2
    Insula_connection.append(CustomLinear(con_matrix1))

    Insula = InsulaNet(Insula_connection)

    a = torch.tensor([[1.,2.,1.,2.]])
    b = torch.tensor([[1., 2., 1., 2.]])
    c = torch.tensor([[2., 2., 4., 2.]])

    confidence = [0, 0]
    for t in range(2):
        Insula(a*10, b*10)
    if sum(sum(Insula.out_Insula)) > 0:
        confidence[0] = confidence[0] + 1
    Insula.reset()

    for t in range(2):
        Insula(a*10, c*10)
    if sum(sum(Insula.out_Insula)) > 0:
        confidence[0] = confidence[0] + 1
    Insula.reset()



    print(confidence)
