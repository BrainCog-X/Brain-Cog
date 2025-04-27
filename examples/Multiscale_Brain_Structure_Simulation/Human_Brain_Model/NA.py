import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
import scipy.io as scio
import pandas as pd
import torch
import networkx as nx
from collections import defaultdict
import community as community_louvain
from matplotlib.ticker import MaxNLocator, FuncFormatter


def histogram_entropy(data, bins='auto'):
    """
    使用直方图法估计一维数据的熵。

    参数:
        data (np.ndarray): 一维数据数组。
        bins (int or str): 直方图的分箱数，默认为 'auto'。

    返回:
        float: 估计的熵值。
    """
    hist, bin_edges = np.histogram(data, bins=bins, density=True)

    bin_width = bin_edges[1] - bin_edges[0]
    prob = hist * bin_width

    prob = prob[prob > 0]

    entropy_value = -np.sum(prob * np.log(prob))

    return entropy_value

def hub_degree(df, W_new):
    degree = torch.sum(W_new, dim=0)
    v, ind = torch.topk(degree, 10)
    ind = ind.tolist()
    plt.figure(figsize=(40, 18))
    plt.bar(df['Identifier'].values, degree)
    plt.bar(df['Identifier'].iloc[ind].values, degree[ind], color='r', label='Top 10 Degree')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=False, prune='lower', nbins=15))
    plt.xticks(rotation=90, fontsize=30)
    plt.ylabel('Degree', fontsize=40)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=40)
    xticks = plt.gca().get_xticklabels()
    for i, tick in enumerate(xticks):
        if df['Identifier'].iloc[i] in df['Identifier'].iloc[ind].values:
            tick.set_color('r')
    plt.grid(axis='y')
    plt.show()

def visual(df, W_new):
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if W_new[i, j] > 0.1:
                ax.plot([x[i], x[j]],
                        [y[i], y[j]],
                        [z[i], z[j]],
                        'k-', lw=1)

    plt.show()

if __name__ ==  "__main__":
    W = np.load('./IIT_connectivity_matrix.npy')
    W = torch.from_numpy(W).float()
    W = W[0:84, 0:84]
    new_order = list(range(0, 35)) + list(range(49, 84)) + list(range(35, 49))
    W_new = W[new_order, :][:, new_order]
    M = torch.max(W_new)
    W_new = W_new / M

    G = nx.from_numpy_matrix(W_new.numpy())

    # Louvain
    partition = community_louvain.best_partition(G)

    community_groups = defaultdict(list)

    for node, community in partition.items():
        community_groups[community].append(node)

    df = pd.read_csv('brain_regions.csv')
    labels = df['Identifier'].values
    # for community, nodes in community_groups.items():
    #     print(f"Community {community}: {nodes}")
    fig, ax = plt.subplots(figsize=(20, 20))
    cax = ax.imshow(W_new.cpu().numpy(), cmap='viridis')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)
    fig.colorbar(cax, shrink=0.8)
    # plt.tight_layout()
    plt.show()
    hub_degree(df, W_new)