# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/9/1 16:43
# User      : Floyed
# Product   : PyCharm
# Project   : BrainCog
# File      : __init__.py.py
# explain   :

import os
import numpy as np
from .genotypes import PRIMITIVES, Genotype

forward_edge_num = sum(1 for i in range(3) for n in range(2 + i))
backward_edge_num = sum(1 for i in range(3) for n in range(i))
num_ops = len(PRIMITIVES)
type_num = len(PRIMITIVES) // 2
# edge_num = [2, 3, 4]

# node_id: (forward) 2, 3, 4
# node_id: (backward) 3, 2
edge_num = [2, 3, 4, 1, 2]


def parse(weights, operation_set,
          op_threshold, parse_method,
          steps, reduction=False,
          back_connection=False):
    global k_best
    gene = []
    if parse_method == 'darts':
        n = 2
        start = 0
        for i in range(steps):  # step = 4
            end = start + n
            W = weights[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -
                           max(W[x][k] for k in range(len(W[x]))))[:2]
            for j in edges:

                for k in range(len(W[j])):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
                # geno item : (operation, node idx)
                gene.append((operation_set[k_best], j))
            start = end
            n += 1

    elif parse_method == 'bio_darts':
        weights_backward = weights[forward_edge_num:]
        weights_forward = weights[:forward_edge_num]

        # forward
        n = 2
        start = 0

        # idx = np.argsort(weights_forward[:, 0]).tolist()
        # if reduction:
        #     idx.remove(0)
        #     idx.remove(1)
        # weights_forward[:, 0] = 0.
        # weights_forward[idx[-2:], 0] = 1.

        for i in range(steps):  # step = 4
            end = start + n
            W = weights_forward[start:end].copy()
            edges = sorted(range(i + 2), key=lambda x: -
                           max(W[x][k] for k in range(len(W[x]))))[:2]
            k_best = None
            idx = np.argsort(W[edges[0]])
            gene.append((operation_set[idx[-1]], edges[0]))
            idx = np.argsort(W[edges[1]])
            gene.append((operation_set[idx[-1]], edges[1]))
            #
            # op_name = operation_set[idx[-1]]
            # idx = np.argsort(W[edges[1]])
            # if 'skip' in op_name:
            #     gene.append((operation_set[idx[-1]], edges[1]))
            # elif '_n' in op_name:
            #     for k in reversed(idx):
            #         if '_n' not in operation_set[k]:
            #             gene.append((operation_set[k], edges[1]))
            #             break
            # else:
            #     for k in reversed(idx):
            #         if '_n' in operation_set[k]:
            #             gene.append((operation_set[k], edges[1]))
            #             break

            start = end
            n += 1

        if back_connection:
            # backward
            n = 1
            start = 0
            for i in range(1, steps):
                end = start + n
                W = weights_backward[start:end].copy()
                edges = sorted(range(i), key=lambda x: -
                               max(W[x][k] for k in range(len(W[x]))))[0]
                idx = np.argsort(W[edges])
                gene.append((operation_set[idx[-1]] + '_back', edges + 2))

                start = end
                n += 1

    elif 'threshold' in parse_method:
        n = 2
        start = 0
        for i in range(steps):  # step = 4
            end = start + n
            W = weights[start:end].copy()
            if 'edge' in parse_method:
                edges = list(range(i + 2))
            else:  # select edges using darts methods
                edges = sorted(range(i + 2), key=lambda x: -
                               max(W[x][k] for k in range(len(W[x]))))[:2]

            for j in edges:
                if 'edge' in parse_method:  # OP_{prob > T} AND |Edge| <= 2
                    topM = sorted(enumerate(W[j]), key=lambda x: x[1])[-2:]
                    for k, v in topM:  # Get top M = 2 operations for one edge
                        if W[j][k] >= op_threshold:
                            gene.append((operation_set[k], i + 2, j))
                # max( OP_{prob > T} ) and |Edge| <= 2
                elif 'sparse' in parse_method:
                    k_best = None
                    for k in range(len(W[j])):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                    if W[j][k_best] >= op_threshold:
                        gene.append((operation_set[k_best], i + 2, j))
                else:
                    raise NotImplementedError(
                        "Not support parse method: {}".format(parse_method))
            start = end
            n += 1
    return gene


def parse_genotype(alphas, steps, multiplier, path=None,
                   parse_method='threshold_sparse', op_threshold=0.85):
    alphas_normal, alphas_reduce = alphas
    gene_normal = parse(alphas_normal, PRIMITIVES,
                        op_threshold, parse_method, steps)
    gene_reduce = parse(alphas_reduce, PRIMITIVES,
                        op_threshold, parse_method, steps)
    concat = range(2 + steps - multiplier, steps + 2)
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat
    )

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        print('Architecture parsing....\n', genotype)
        save_path = os.path.join(
            path, parse_method + '_' + str(op_threshold) + '.txt')
        with open(save_path, "w+") as f:
            f.write(str(genotype))
            print('Save in :', save_path)

