import time
import scipy.io as scio
import torch
from braincog.base.node.node import *
from braincog.base.brainarea.BrainArea import *
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt


class brain_region(BrainArea):
    def __init__(self, name, num_neuron, neuron_type_ratio, neuron_type_p, neuron_type_index, W):
        """
        num_neuron: neuron number of this brain region
        neuron_type_ratio: the ratio of different neuron type in this brain region
        neuron_type_p: the neuron parameter of different neuron type in this brain region
        W: connection weight for different neuron type
        neuron_type_index: neuron type index of the neuron type in this brain region
        """
        super(brain_region, self).__init__()
        self.name = name
        self.EX = []  # excitatory neuron index set
        self.IN = []  # inhibitory neuron index set
        for i in range(len(neuron_type_p)):
            if neuron_type_p[i][6]:
                self.IN.append(i)
            else:
                self.EX.append(i)
        self.W = W[neuron_type_index]
        self.num_neuron_per_type = torch.tensor([int(num_neuron * neuron_type_ratio[i])
                                                 for i in range(len(neuron_type_ratio))])
        if neuron_model == 'aEIF':
            self.neuron = [aEIF(neuron_type_p[i], self.num_neuron_per_type[i], self.W[i][neuron_type_index], i)
                           for i in range(len(neuron_type_ratio))]
        if neuron_model == 'HH':
            self.neuron = [HHNode(neuron_type_p[i], self.num_neuron_per_type[i], self.W[i][neuron_type_index], i)
                           for i in range(len(neuron_type_ratio))]
        self.num_neuron = torch.sum(self.num_neuron_per_type)
        # the index of neuron who receive the input from brain regions
        self.input_neuron = [torch.randint(0, self.num_neuron_per_type[i],
                                           (1, int(self.num_neuron_per_type[i] * 0.1) + 1)).squeeze(0)
                             for i in range(len(neuron_type_ratio))]
        # the index of output neuron
        self.output_neuron = [torch.randint(0, self.num_neuron_per_type[i],
                                            (1, int(self.num_neuron_per_type[i] * 0.1) + 1)).squeeze(0)
                              for i in range(len(neuron_type_ratio))]

        self.spike = [self.neuron[i].spike for i in range(len(neuron_type_ratio))]
        self.mem = [self.neuron[i].mem for i in range(len(neuron_type_ratio))]
        self.internal_connection = []
        self.init_internal_connection()

    def init_internal_connection(self):
        for i in range(len(self.neuron)):  # input connection of neuron type i
            connection = []
            for j in range(len(self.neuron)):  # connection from neuron type j

                connection.append([torch.randint(0, self.num_neuron_per_type[j],
                                                 (1, int(self.num_neuron_per_type[j] * 0.1) + 1)).squeeze(0)
                                   for _ in range(self.num_neuron_per_type[i])])
                # continue

            self.internal_connection.append(connection)

    def get_output_fire_rate(self):
        fire_rate = []
        for j in range(len(self.neuron)):
            fire_rate.append(torch.mean(self.spike[j][self.output_neuron[j]]))
        return fire_rate

    def cal_current(self, neuron, connection, input_neuron, external_input):
        """
        connection[i][j] contains an array of index, the index means the neuron of group i connect to neuron j in
        this group
        spike[i] and mem[i] are the brain region's i-th neuron group's spike and membrane at the last
        IN means the inhibitory neuron group's index set
        input_neuron are the neuron who receive the input from other brain region
        external_input are the input from other brain region
        """

        neuron.Iback = neuron.dt_over_tau * (torch.randn(neuron.neuron_num) - neuron.Iback)
        neuron.Ieff = neuron.Iback / neuron.sqrt_coeff * neuron.sig + neuron.mu
        # #
        for j in range(neuron.neuron_num):  # j-th neuron in the group
            dIchem = -neuron.Ichem[j]
            for i in range(len(neuron.W)):
                dIchem = dIchem + neuron.W[i] * torch.mean(self.spike[i][connection[i][j]])
            if j in input_neuron:
                dIchem = dIchem + external_input
            neuron.Ichem[j] = neuron.Ichem[j] + neuron.dt_over_tau * dIchem
            if neuron.if_IN:
                Vgap = -self.mem[neuron.type_index][j]
                for i in self.IN:
                    Vgap = Vgap + torch.mean(self.mem[i][connection[i][j]])
                neuron.Igap[j] = neuron.Gama_c * Vgap
        if neuron.if_IN:
            current = neuron.Ieff + neuron.Ichem + neuron.Igap

        else:
            current = neuron.Ieff + neuron.Ichem
        # current = neuron.Ieff
        return current

    def forward(self, external_input):

        for i in range(len(self.neuron)):
            external_input_i = torch.sum(self.W[i] * external_input)
            # print(external_input)
            current = self.cal_current(self.neuron[i], self.internal_connection[i],
                                       self.input_neuron[i], external_input_i)
            self.neuron[i](current)
        fire_rate = []
        for i in range(len(self.neuron)):
            self.spike[i] = self.neuron[i].spike
            self.mem[i] = self.neuron[i].mem
            fire_rate.append(torch.mean(self.spike[i]))
        print(f'neuron group fire-rate of region {self.name}:\n'
              f'{fire_rate}')

        return self.spike, self.mem


if __name__ == '__main__':
    weight_matrix = torch.tensor(genfromtxt("./human.csv", delimiter=',', skip_header=False)) / 100
    weight_matrix = weight_matrix.to(torch.float32)
    neuron_model = 'aEIF'
    if neuron_model == 'aEIF':
        # p: [threshold, c_m, alpha_w, beta_ad, mu, sig, if_IN]
        p_E = [5, 1, 0, 0, 20, 12, 0]
        p_I_BC = [4, 1, -2, 4.5, 5, 10, 1]
        p_I_MC = [4, 0.8, -2, 4.5, 5, 10, 1]
        p_TC = [5, 2, 0, 0, 20, 12, 0]
        p_TI = [4, 0.2, -2, 4.5, 5, 10, 1]
        p_TRN = [4, 0.4, -2, 4.5, 5, 10, 1]
    if neuron_model == 'HH':
        p_E = [3, 2, 0.3, 10, -2, 10, 1, 5, 0]
        p_I_BC = [2, 1.5, 0.3, 12, -1, 20, 2, 6, 1]
        p_I_MC = [2, 1.5, 0.3, 12, -1, 20, 2, 6, 1]
        p_TC = [3, 2, 0.3, 12, -2, 10, 1, 5, 0]
        p_TI = [2, 1.5, 0.3, 12, -1, 20, 2, 6, 1]
        p_TRN = [3, 1.5, 0.3, 12, -1, 20, 2, 6, 1]
    CR_neuron_ratio = [0.7, 0.2, 0.1]
    CR_neuron_p = [p_E, p_I_BC, p_I_MC]
    TN_neuron_ratio = [0.6, 0.15, 0.25]
    TN_neuron_p = [p_TC, p_TI, p_TRN]

    # connection weight between neuron type
    #                  E, I_BC, I_MC, TC, TI, TRN
    W = torch.tensor([[30, -20, -20, 30, -20, -20],  # E
                      [30, 30, 30, 30, 30, 30],  # I_BC
                      [30, 30, 30, 30, 30, 30],  # I_MC
                      [30, -20, -20, 30, -20, -20],  # TC
                      [30, 30, 30, 30, 30, 30],  # TI
                      [30, 30, 30, 30, 30, 30]])  # TRN

    NR = len(weight_matrix)
    NCR = 210
    NTN = 36
    # NR = 25
    # NCR = 20
    # NTN = 5
    # Scale down proportionally, if scale=1, the model has about 50341200 neurons in total
    # the range of parameter scale is 0.01 <= scale <= 1
    scale = 0.01
    T = 40
    start = time.time()
    # initialize brain regions
    brain_regions = []

    for i in range(NCR):
        brain_regions.append(brain_region(f'{i}', 3000,
                                          CR_neuron_ratio, CR_neuron_p, [0, 1, 2], W))

    for i in range(NCR, NCR + NTN):
        brain_regions.append(brain_region(f'{i}', 1000,
                                          TN_neuron_ratio, TN_neuron_p, [3, 4, 5], W))

    end = time.time()
    ms = (end - start) * 10 ** 3
    print(f"spend {ms:.03f} ms.")
    print('Finish initialization')

    CR_Matrix = weight_matrix[:, 0:NCR]
    TN_Matrix = weight_matrix[:, NCR:NCR + NTN]

    output_E = torch.zeros(NCR)
    output_I_BC = torch.zeros(NCR)
    output_I_MC = torch.zeros(NCR)

    output_TC = torch.zeros(NTN)
    output_TI = torch.zeros(NTN)
    output_TRN = torch.zeros(NTN)
    Iraster = []

    # start stimulation
    for t in range(T):
        start = time.time()
        # fire-rate of neuron groups of different brain regions
        for n in range(NCR):
            output = brain_regions[n].get_output_fire_rate()
            output_E[n] = output[0]
            output_I_BC[n] = output[1]
            output_I_MC[n] = output[2]

        for n in range(NCR, NCR + NTN):
            output = brain_regions[n].get_output_fire_rate()
            output_TC[n - NCR] = output[0]
            output_TI[n - NCR] = output[1]
            output_TRN[n - NCR] = output[2]
        # cross-brain-region input
        input_E = torch.matmul(CR_Matrix, output_E)
        input_I_BC = torch.matmul(CR_Matrix, output_I_BC)
        input_I_MC = torch.matmul(CR_Matrix, output_I_MC)

        input_TC = torch.matmul(TN_Matrix, output_TC)
        input_TI = torch.matmul(TN_Matrix, output_TI)
        input_TRN = torch.matmul(TN_Matrix, output_TRN)

        spike = []
        for n in range(NR):
            external_input = torch.tensor([input_E[n], input_I_BC[n], input_I_MC[n],
                                           input_TC[n], input_TI[n], input_TRN[n]])
            brain_regions[n](external_input)
            spike.extend(brain_regions[n].spike)

        spike = torch.concatenate(spike)
        Isp = torch.nonzero(spike)
        print(len(Isp))
        if (len(Isp) != 0):
            left = t * torch.ones((len(Isp)))
            left = left.reshape(len(left), 1)
            mide = torch.concatenate((left, Isp), dim=1)
        if (len(Isp) != 0) and (len(Iraster) != 0):
            Iraster = torch.concatenate((Iraster, mide), dim=0)
            print('here')
        if (len(Iraster) == 0) and (len(Isp) != 0):
            Iraster = mide
            print('first')

        print(t)
        end = time.time()
        ms = (end - start) * 10 ** 3
        print(f"Elapsed {ms:.03f} ms.")

    Iraster = torch.tensor(Iraster).transpose(0, 1)
    torch.save(Iraster, "./human2.pt")
    plt.figure(figsize=(10, 10))
    plt.scatter(Iraster[0], Iraster[1], c='k', marker='.', s=0.001)
    plt.savefig('human2.jpg')
    plt.show()
