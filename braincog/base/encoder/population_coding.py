import torch
import torch.nn as nn
import torchvision.utils

class PEncoder(nn.Module):
    """
    Population coding
    :param step: time steps
    :param encode_type: encoder type (str)
    """
    def __init__(self, step, encode_type):
        super().__init__()
        self.step = step
        self.fun = getattr(self, encode_type)

    def forward(self, inputs, num_popneurons, *args, **kwargs):
        outputs = self.fun(inputs, num_popneurons, *args, **kwargs)
        return outputs

    @torch.no_grad()
    def population_time(self, inputs, m):
        """
        one feature will be encoded into gauss_neurons
        the center of i-th neuron is:  gauss --

        .. math::
            \\mu  u_i = I_min + (2i-3)/2(I_max-I_min)/(m -2)
        the width of i-th neuron is :  gauss --

        .. math::
            \\sigma sigma_i = \\frac{1}{1.5}\\frac{(I_max-I_min)}{m - 2}

        :param inputs:   (N_num, N_feature) array
        :param m: the number of the gaussian neurons
        i : the i_th gauss_neuron
        1.5: experience value
        popneurons_spike_t: gauss -- function
        I_min = min(inputs)
        I_max = max(inputs)
        :return: (step, num_gauss_neuron) 
        """
        # m = self.step
        I_min, I_max = torch.min(inputs), torch.max(inputs)
        mu = [i for i in range(0, m)]
        mu = torch.ones((1, m)) * I_min + ((2 * torch.tensor(mu) - 3) / 2) * ((I_max-I_min) / (m -2))
        sigma = (1 / 1.5) * ((I_max-I_min) / (m -2))
        # shape = (self.step,) + inputs.shape
        shape = (self.step,m)
        popneurons_spike_t = torch.zeros(((m,) + inputs.shape))
        for i in range(m):
            popneurons_spike_t[i, :] = torch.exp(-(inputs - mu[0, i]) ** 2 / (2 * sigma * sigma))

        spike_time = (self.step * popneurons_spike_t).type(torch.int)
        spikes = torch.zeros(shape)
        for spike_time_k in range(self.step):
            if torch.where(spike_time == spike_time_k)[1].numel() != 0:
                spikes[spike_time_k][torch.where(spike_time == spike_time_k)[0]] = 1

        return spikes

    @torch.no_grad()
    def population_voltage(self, inputs, m, VTH):
        '''
        The more similar the input is to the mean,
        the more sensitive the neuron corresponding to the mean is to the input.
        You can change the maen.
        :param inputs:   (N_num, N_feature) array
        :param m : the number of the gaussian neurons
        :param VTH : threshold voltage
        i : the i_th gauss_neuron
        one feature will be encoded into gauss_neurons
        the center of i-th neuron is:  gauss -- \mu  u_i = I_min + (2i-3)/2(I_max-I_min)/(m -2)
        the width of i-th neuron is :  gauss -- \sigma sigma_i = 1/1.5(I_max-I_min)/(m -2) 1.5: experience value
        popneuron_v: gauss -- function
        I_min = min(inputs)
        I_max = max(inputs)
        :return: (step, num_gauss_neuron, dim_inputs) 
        '''
        ENCODER_REGULAR_VTH = VTH
        I_min, I_max = torch.min(inputs), torch.max(inputs)
        mu = [i for i in range(0, m)]
        mu = torch.ones((1, m)) * I_min + ((2 * torch.tensor(mu) - 3) / 2) * ((I_max-I_min) / (m -2))
        sigma = (1 / 1.5) * ((I_max-I_min) / (m -2))
        popneuron_v = torch.zeros(((m,) + inputs.shape))
        delta_v = torch.zeros(((m,) + inputs.shape))
        for i in range(m):
            delta_v[i] = torch.exp(-(inputs - mu[0, i]) ** 2 / (2 * sigma * sigma))
        spikes = torch.zeros((self.step,) + ((m,) + inputs.shape))
        for spike_time_k in range(self.step):
            popneuron_v = popneuron_v + delta_v
            spikes[spike_time_k][torch.where(popneuron_v.ge(ENCODER_REGULAR_VTH))] = 1
            popneuron_v = popneuron_v - spikes[spike_time_k] * ENCODER_REGULAR_VTH

        popneuron_rate = torch.sum(spikes, dim=0)/self.step

        return spikes, popneuron_rate


## test
# if __name__ == '__main__':
#     a = (torch.rand((2,4))*10).type(torch.int)
#     print(a)
#     pencoder = PEncoder(10, 'population_time')
#     spikes=pencoder(inputs=a, num_popneurons=3)
#     print(spikes, spikes.shape)

#     pencoder = PEncoder(10, 'population_voltage')
#     spikes, popneuron_rate = pencoder(inputs=a, num_popneurons=5, VTH=0.99)
#     print(spikes, spikes.shape)
