import matplotlib.colors as mcolors
import torch
import random
import sys
from random import sample
from matplotlib import pyplot as plt
from tools.MazeTurnEnvVec import *
import math
import networkx as nx
np.set_printoptions(precision = 3)
from LSM_LIF import LSM

class population(object):
    def __init__(self, matrix):
        self.pop=[]
        for i in matrix:
            m = LSM(n_offsprings=1)
            m.liquid_weight_matrix=i
            self.pop.append(m)


def one_hot_compute_rank(model):
    all_state=np.array([[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0]])
    all_state*=2
    n_states=all_state.shape[0]
    offs=model.liquid_weight_matrix
    r_matrix=torch.zeros([n_states,model.n_offsprings,model.num_of_liquid_layer_neurons])
    for i in range(n_states):
        ii=np.tile(all_state[i],model.n_offsprings).reshape([model.n_offsprings,model.n_input])
        model.predict_on_batch(ii)
        r_matrix[i]=model.liquid_s_list
        draw_spikes(model,inputsize=10,id=0, i_s=ii, l_s=model.sumspikes[0], r_s=model.sumspikes[1])
    r_matrix=r_matrix.permute(1,0,2)
    r=torch.linalg.matrix_rank(r_matrix)
    r_matrix = r_matrix.permute(0, 2, 1)

    return r

def compute_rank(off):
    all_state=np.array([[0,1,1,1],[1,0,0,0],[1,2,1,0],[1,0,1,1],[0,1,0,0],[1,1,0,1],[2,1,2,1]])
    all_state*=3
    n_states=all_state.shape[0]
    r_matrix=torch.zeros([n_states,off.n_offsprings,off.num_of_liquid_layer_neurons])
    for i in range(n_states):
        ii=np.tile(all_state[i],off.n_offsprings).reshape([off.n_offsprings,off.n_input])
        off.predict_on_batch(ii)
        r_matrix[i]=off.liquid_s_list
    r_matrix=r_matrix.permute(1,0,2)
    r=torch.linalg.matrix_rank(r_matrix)
    r_matrix = r_matrix.permute(0, 2, 1)
    return r
    
def draw_spikes(model, inputsize,id, l_s, r_s,i_s):
    m = model.liquid_weight_matrix[id].cpu().numpy()
    if type(i_s)==int:
        i_s=np.zeros(inputsize)
    elif (torch.is_tensor(i_s) == True):
        i_s = i_s[id].cpu().numpy()
    else:
        i_s = i_s[id]

    neurons = model.num_of_liquid_layer_neurons
    num = model.width
    #
    D = nx.Graph(m)

    pos = []
    i = 0
    ######################## liquid neurons' position
    for n in D.nodes:
        pos.append([i // num, i % num])
        i += 1
    node_spikes = l_s[id].cpu().numpy()
    ######################## add input nodes, 64,65,66
    for j in range(model.n_input):
        D.add_node(j + neurons)
        pos.append([-3, j])
    neurons = neurons + model.n_input
    node_spikes = np.concatenate((node_spikes, i_s), axis=0)
    ####################### add output nodes, 67,68,69
    for j in range(model.n_output):
        D.add_node(j + neurons)
        pos.append([num + 2, j])

    outnodesipkes = r_s[id].cpu().numpy()
    node_spikes = np.concatenate((node_spikes, outnodesipkes), axis=0)
    D.add_weighted_edges_from(model.input_to_primary_list[id])
    D.add_weighted_edges_from(model.liquid_to_output_list[id])
    for u,v,d in D.edges(data=True):
        d['weight']=round(d['weight'],2)
    edges, weights = zip(*nx.get_edge_attributes(D, 'weight').items())

    v = list(i for i in range(0, neurons + model.n_output))
    # node_label = dict(zip(v, list(np.array(D.degree)[:, 1])))

    node_color = np.around(node_spikes, 3)
    # node_color=np.array(np.array(node_color, dtype=bool), dtype=int)
    node_label = dict(zip(v, list(node_color)))

    cmap1 = plt.cm.RdBu_r
    bins = np.array([-1, -0.8, -0.5, -0.1, 0.1, 0.5, 0.8, 1])
    nbin = len(bins) - 1
    n_negative = np.count_nonzero(bins < 0)
    n_positive = np.count_nonzero(bins > 0)
    colors = np.vstack((
        cmap1(np.linspace(0, 0.5, n_negative))[:-1],
        cmap1(np.linspace(0.5, 1, n_positive))
    ))  # 根据bins的区间数新建colormap.
    cmap2 = mcolors.ListedColormap(colors)
    cmap = plt.cm.get_cmap('Blues')

    # print(np.std(np.array(weights)))
    # plt.figure(figsize=(18,17))
    #
    if max(node_color) <= 3:
        vm = 3
    else:
        vm = 20
    nx.draw(D, pos=pos, edge_color=weights, edge_cmap=cmap, node_color=node_color, cmap=cmap, vmin=0, vmax=vm,
            edge_vmin=-1, edge_vmax=2)

    nx.draw_networkx_labels(D,pos=pos,labels=node_label)
    edge_labels=nx.get_edge_attributes(D, 'weight')
    # nx.draw_networkx_edge_labels(D,pos=pos,edge_labels=edge_labels,font_size=6)

    plt.show()

def update_matrix_to_list(matrix, id):
    spikes_list = []
    for i in range(matrix.size()[0]):
        small_list = []
        none_zero = torch.nonzero(matrix[i]).cpu().numpy()
        m = matrix[i].cpu().numpy()
        if none_zero.size != 0:
            for j in none_zero:
                edge = [j[0] + id, j[1], m[j[0], j[1]]]
                small_list.append(edge)
        spikes_list.append(small_list)
    return spikes_list

def calc_priority_based_on_dis(num,size,w_scale=0.8,neighbors=2,l=2):
    p=np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            xi=i//size
            xj=j//size
            yi=i%size
            yj=j%size
            dx=abs(xi-xj)
            dy=abs(yi-yj)
            if dx>neighbors or dy>neighbors or dx+dy==0:
                p[i,j]=0
            else:
                p[i,j]=math.exp((-dx**2-dy**2)/l**2)
    return p*w_scale

