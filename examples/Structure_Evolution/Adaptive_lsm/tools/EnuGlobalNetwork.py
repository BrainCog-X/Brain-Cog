import pickle
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from AbstractLayerBMM import AbstractLayerBMM
from EvolvableNeuralUnitStacked import EvolvableNeuralUnitStacked
from Tools import get_data_path

sns.set_style("darkgrid")


class EnuGlobalNetwork(AbstractLayerBMM):
    """Network of ENUs implementation in PyTorch, where each synapse and neuron is modeled as an ENU. """

    def __init__(self, n_offspring, n_pseudo_env, n_input_neurons, n_hidden_neurons, n_output_neurons, n_syn_per_neuron):
        # offspring
        self.n_offspring = n_offspring
        self.n_pseudo_env = n_pseudo_env
        # input channels
        n_input_channels = 16
        self.n_input_channels = n_input_channels
        n_dynamic_param = 32
        # total neurons
        n_neurons = n_output_neurons + n_hidden_neurons
        self.n_neurons = n_neurons
        super().__init__(n_offspring, n_neurons, n_input_neurons, n_output_neurons)
        torch.random.manual_seed(0)
        #NOTE: batch dimension holds output of each neuron/synapse, allowing fast GPU MM
        #NOTE neurons far less than synapses, so can be relatively bigger rnn for little cost
        n_input_channels_neuron = 16
        n_input_neuron, n_output_neuron = n_input_channels_neuron, n_input_channels
        self.neurons = EvolvableNeuralUnitStacked(n_offspring, batch_size=self.n_neurons, n_input=n_input_neuron, n_dynamic_param=n_dynamic_param, n_output=n_output_neuron)
        #self.n_syn = next_power_of_2(int(n_neurons * (rel_connectivity*n_neurons)))
        self.n_syn_per_neuron = n_syn_per_neuron
        self.n_syn = n_neurons * n_syn_per_neuron
        n_input_syn, n_output_syn = n_input_channels * 2, n_input_channels_neuron # * 2 for neuron feedback (which same channel as n_channel input)
        self.synapses = EvolvableNeuralUnitStacked(n_offspring, batch_size=self.n_syn, n_input=n_input_syn, n_dynamic_param=n_dynamic_param, n_output=n_output_syn)
        # just randomly connect synapses to neurons
        self.synapse_connections = torch.randint(n_input_neurons + n_neurons, size=(n_neurons, n_syn_per_neuron), device='cuda', dtype=torch.long)
        # fixed predefined connection patterns
        if n_input_neurons==2 and n_output_neurons==2 and n_hidden_neurons==2:
            print("Fixed connection Network 2-2-2")
            self.synapse_connections = torch.tensor([[0, 1],
                                                     [0, 1],
                                                     [2, 3],
                                                     [2, 3]], device='cuda', dtype=torch.long)
        elif n_input_neurons == 4 and n_output_neurons == 3 and n_hidden_neurons == 3 and n_syn_per_neuron==3:
            print("Fixed connection Network 4-3-3 (3syn)")
            self.synapse_connections = torch.tensor([[0, 1, 3],# hidden connections #4
                                                     [0, 2, 3], #5
                                                     [1, 2, 3],# 6
                                                     [4, 5, 6], # output connections #7
                                                     [4, 5, 6],#8
                                                     [4, 5, 6]#9
                                                     ], device='cuda', dtype=torch.long)
        elif n_input_neurons==5 and n_hidden_neurons==0 and n_output_neurons==4:
            print("Fixed connection Network 5-0-4 (5syn)")
            # neuron i connected to neuron j and k, neuron 0..input_neurons is index
            self.synapse_connections = torch.tensor([[0, 1, 2, 3, 4],# output connections
                                                     [0, 1, 2, 3, 4],
                                                     [0, 1, 2, 3, 4],
                                                     [0, 1, 2, 3, 4]
                                                     ], device='cuda', dtype=torch.long)
        elif n_input_neurons==1 and n_hidden_neurons==0 and n_output_neurons==2:
            print("Fixed connection Network 1-0-2 (1syn)")
            # neuron i connected to neuron j and k, neuron 0..input_neurons is index
            self.synapse_connections = torch.tensor([[0],# output connections
                                                     [0]
                                                     ], device='cuda', dtype=torch.long)
        elif n_input_neurons==4 and n_hidden_neurons==0 and n_output_neurons==3 and n_syn_per_neuron==4:
            print("Sparse connection Network 4-0-3 (4syn)")
            # neuron i connected to neuron j and k, neuron 0..input_neurons is index
            self.synapse_connections = torch.tensor([[0, 1, 2, 3],# output connections #4
                                                     [0, 1, 2, 3], #5
                                                     [0, 1, 2, 3],# 6
                                                     ], device='cuda', dtype=torch.long)
        elif n_input_neurons == 4 and n_hidden_neurons == 3 and n_output_neurons == 3 and n_syn_per_neuron == 4:
            print("Sparse connection Network 4-3-3 (3syn)")
            # neuron i connected to neuron j and k, neuron 0..input_neurons is index
            self.synapse_connections = torch.tensor([[0, 1, 3],  # hidden connections #4
                                                     [0, 2, 3],  # 5
                                                     [1, 2, 3],  # 6
                                                     [4, 5, 3],  # output connections #7
                                                     [4, 6, 3],  # 8
                                                     [5, 6, 3]  # 9
                                                     ], device='cuda', dtype=torch.long)
        elif n_input_neurons==4 and n_hidden_neurons==3 and n_output_neurons==3 and n_syn_per_neuron==8:
            print("Sparse connection Network 4-3-3 (8syn)")
            # neuron i connected to neuron j and k, neuron 0..input_neurons is index
            self.synapse_connections = torch.tensor([[0, 1, 5, 6, 7, 8, 3, 4],# hidden connections #4
                                                     [0, 2, 4, 6, 7, 9, 3, 5], #5
                                                     [1, 2, 4, 5, 8, 9, 3, 6],# 6
                                                     [4, 5, 8, 9, 0, 1, 3, 7], # output connections #7
                                                     [4, 6, 7, 9, 0, 2, 3, 8],#8
                                                     [5, 6, 7, 8, 1, 2, 3, 9]#9
                                                     ], device='cuda', dtype=torch.long)
        elif n_input_neurons==4 and n_hidden_neurons==4 and n_output_neurons==4 and n_syn_per_neuron==8:
            print("Fixed connection Network 4-4-4 (8syn)")
            # neuron i connected to neuron j and k, neuron 0..input_neurons is index
            self.synapse_connections = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],# hidden connections
                                                     [0, 1, 2, 3, 4, 5, 6, 7],
                                                     [0, 1, 2, 3, 4, 5, 6, 7],
                                                     [0, 1, 2, 3, 4, 5, 6, 7],
                                                     [4, 5, 6, 7, 8, 9, 10, 11], # output connections
                                                     [4, 5, 6, 7, 8, 9, 10, 11],
                                                     [4, 5, 6, 7, 8, 9, 10, 11],
                                                     [4, 5, 6, 7, 8, 9, 10, 11],
                                                     ], device='cuda', dtype=torch.long)
        elif n_input_neurons==5 and n_hidden_neurons==5 and n_output_neurons==4:
            print("Fixed connection Network 5-5-4 (5syn)")
            # neuron i connected to neuron j and k, neuron 0..input_neurons is index
            self.synapse_connections = torch.tensor([[0, 1, 2, 3, 4],# hidden connections
                                                     [0, 1, 2, 3, 4],
                                                     [0, 1, 2, 3, 4],
                                                     [0, 1, 2, 3, 4],
                                                     [0, 1, 2, 3, 4],
                                                     [5, 6, 7, 8, 9], # output connections
                                                     [5, 6, 7, 8, 9],
                                                     [5, 6, 7, 8, 9],
                                                     [5, 6, 7, 8, 9],
                                                     ], device='cuda', dtype=torch.long)
        elif n_input_neurons==1 and n_hidden_neurons==0 and n_output_neurons==1:
            print("Fixed connection Single")
            self.synapse_connections = torch.tensor([[0]], device='cuda', dtype=torch.long)
        else:
            print("Random connections")
        # each synapse is connected also to its post-synaptic neuron, to allow STDP type learning to emerge
        self.synapse_connections_post = torch.arange(n_neurons, device='cuda', dtype=torch.long).reshape(n_neurons, -1).repeat(1, n_syn_per_neuron)
        # define compartments
        self.compartments = [self.neurons, self.synapses]
        self.trainable_layers = self.neurons.trainable_layers + self.synapses.trainable_layers
        self.track_data = False


    def dump_model(self, e, exp_name):
        """Dump model to restore"""
        with open(get_data_path(e, exp_name, "Model"), 'wb') as f:
            parameters = {}
            parameters["neuron"] = [layer.base_parameters.cpu().numpy() for layer in self.neurons.trainable_layers]
            parameters["synapse"] = [layer.base_parameters.cpu().numpy() for layer in self.synapses.trainable_layers]
            pickle.dump(parameters, f)

    def restore_model(self, e, exp_name):
        """Restore model"""
        with open(get_data_path(e, exp_name, "Model"), 'rb') as f:
            parameters = pickle.load(f)
        #TODO: refactor to dump/restore at ENU level and just call those functions
        assert len(self.neurons.trainable_layers) == len(parameters["neuron"])
        for i in range(len(parameters["neuron"])):
            self.neurons.trainable_layers[i].base_parameters = torch.from_numpy(parameters["neuron"][i].astype(np.float32)).cuda()
        assert len(self.synapses.trainable_layers) == len(parameters["synapse"])
        for i in range(len(parameters["synapse"])):
            self.synapses.trainable_layers[i].base_parameters = torch.from_numpy(parameters["synapse"][i].astype(np.float32)).cuda()

    @staticmethod
    def plot_weights(e, exp_name):
        """Visualize weights of ENU gates"""
        sns.set_style("dark")
        def calc_average(start, stop):
            weights_average = None
            for e in range(start, stop, 1000):
                with open(get_data_path(e, exp_name, "Model"), 'rb') as f:
                    parameters = pickle.load(f)
                weights = []
                for i in range(len(parameters["neuron"])):
                    weights += [parameters["neuron"][i].astype(np.float32)]
                if weights_average is None:
                    weights_average = weights
                else:
                    for i in range(len(weights_average)):
                        weights_average[i] += weights[i]
            return weights_average
        weights_mean1 = calc_average(20000, 30000)
        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
        for i in range(len(weights_mean1)):
            ax[i].imshow(weights_mean1[i], cmap="gray")
        weights_mean2 = calc_average(30000, 40000)
        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
        for i in range(len(weights_mean2)):
            ax[i].imshow(weights_mean2[i], cmap="gray")
        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
        for i in range(len(weights_mean2)):
            ax[i].imshow((weights_mean2[i] - weights_mean1[i])**5, cmap="gray")
        plt.show()


    def dump_network_activity(self, e, exp_name):
        """Dump raw data for visualization"""
        with open(get_data_path(e, exp_name, "GlobalNetwork"), 'wb') as f:
            pickle.dump(self.vis_data, f)

    def print(self):
        print("--Neurons--")
        self.neurons.print()
        print("--Synapses--")
        self.synapses.print()

    def reset(self):
        self.vis_data = []
        if self.track_data:
            print("Tracking network activity")
        for compartment in self.compartments:
            compartment.reset()

    def forward(self, X):
        """Main computation forward pass"""
        # transfer to GPU
        X_raw_gpu = torch.from_numpy(X.astype(np.float32)).cuda()
        X_gpu = torch.zeros((X.shape[0], X.shape[1], self.n_input_channels), device='cuda', dtype=torch.float32)
        X_gpu[:, :, :X_raw_gpu.shape[2]] = X_raw_gpu
        # first compute synapses, set input to previous output of connected neuron
        # concat our input spiking pattern directly to input to our synapses (the neurons)
        # NOTE: this concats in batch dimension, meaning it feeds into input neurons directly spiking pattern, while rest receive input from network
        input_to_synapses = torch.cat([X_gpu, self.neurons.out_mem], dim=1)
        # connect each synapse randomly to multiple inputs
        input_to_synapses_connected = input_to_synapses[:, self.synapse_connections.flatten(), :]
        # need feedback connection from neuron to synapse, to allow stdp type rules to emerge (else it has to do it through feedback connections, but less guarentee on connections and cannot distinguise type)
        # one synapse has 1 pre-synaptic neuron and 1 post-synaptic neuron, connectection defined in synapse_connections, synapse_connections[i, :] gives all input synapses of that neuron
        # so feedback to all it's input synapses through broadcasting backwards
        post_neuron_backprop_connected = self.neurons.out_mem[:, self.synapse_connections_post.flatten(), :]
        input_to_synapses_connected = torch.cat([input_to_synapses_connected, post_neuron_backprop_connected], dim=-1)
        # compute synapse
        self.synapses.forward(input_to_synapses_connected)
        # then integrate(sum) all outputs of a neurons input synapses, can just reshape into valid shape, since we already randomly connected when computing synapses
        # NOTE: each neuron then requires same number of synapses, then reshape by modifying batch dim (which contains syn outputs)
        integration = torch.sum(self.synapses.out.reshape((self.n_offspring, self.n_neurons, -1, self.synapses.shape[-1])), dim=2)
        # scale by number of synapses
        integration /= self.n_syn_per_neuron
        self.out_integration = integration
        # finally set neuron input to summated connected synapses output
        input_to_neurons = integration
        out = self.neurons.forward(input_to_neurons)
        # output is last neuron output, NOTE: just first channel is returned, since we reshape neurons to channels
        self.out = out[:, -self.n_output:, 0].reshape(self.n_offspring, self.n_output)
        if self.track_data:
            self._track_vis_data(X, input_to_synapses_connected, input_to_neurons)
        return self.out

    def _track_vis_data(self, X, input_to_synapses_connected, input_to_neurons):
        offspring_idx = 0
        self.vis_data += [(X[offspring_idx], input_to_neurons[offspring_idx].cpu().numpy(), self.neurons.out[offspring_idx].cpu().numpy(),
                           input_to_synapses_connected[offspring_idx].cpu().numpy(), self.synapses.out[offspring_idx].cpu().numpy())]

    @staticmethod
    def plot_network_activity(e, exp_name):
        with open(get_data_path(e, exp_name, "GlobalNetwork"), 'rb') as f:
            vis_data = pickle.load(f)
            X, input_to_neurons, neurons_out, input_to_synapses, synapses_out = map(np.array, zip(*vis_data))
        def plot_enu_activity(input, output, title):
            n_cells = output.shape[1]
            n_cells = np.minimum(10, output.shape[1])
            fig, grid = plt.subplots(2, n_cells, sharex='col', sharey='row')
            if n_cells==1:
                grid[0].plot(input[:, 0, :])
                grid[1].plot(output[:, 0, :])
            else:
                for i in range(n_cells):
                    grid[0, i].plot(input[:, i, :])
                    grid[1, i].plot(output[:, i, :])
            plt.xlabel("t")
            plt.title(title)
            #plt.ylabel("")
            plt.legend()
        plt.figure()
        plt.plot(X[:, :, 0])
        plot_enu_activity(input_to_neurons, neurons_out, "ENU neuron activity")
        plot_enu_activity(input_to_synapses, synapses_out, "ENU synapse activity")

        plt.figure()
        spike_points = np.where(neurons_out[:, :, 0] > 0)
        plt.scatter(spike_points[0], spike_points[1], marker='|')

        plt.show()


