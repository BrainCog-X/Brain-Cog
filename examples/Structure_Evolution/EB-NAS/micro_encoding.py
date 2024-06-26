# NASNet Search Space https://arxiv.org/pdf/1707.07012.pdf
# code modified from DARTS https://github.com/quark0/darts
import numpy as np
from collections import namedtuple
from random import choice
from numpy.linalg import matrix_rank
import itertools
import torch
# from models.micro_models import NetworkCIFAR as Network

import motifs

# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
# Genotype_norm = namedtuple('Genotype', 'normal normal_concat')
# Genotype_redu = namedtuple('Genotype', 'reduce reduce_concat')
Genotype = namedtuple('Genotype', 'normal normal_concat')

# what you want to search should be defined here and in micro_operations

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'sep_conv_7x7',
    'conv_7x1_1x7',
]
OPERATIONS_back = [
    # 'max_pool_3x3_p_back',
    # 'avg_pool_3x3_p_back',
    'conv_3x3_p_back',
    'conv_5x5_p_back',
    # 'avg_pool_3x3_n_back',
    'conv_3x3_n_back',
    'conv_5x5_n_back',
    # 'sep_conv_3x3_p_back',
    # 'sep_conv_5x5_p_back',
    # 'dil_conv_3x3_p_back',
    # 'dil_conv_5x5_p_back',
    # 'def_conv_3x3_p_back',
    # 'def_conv_5x5_p_back',
]
OPERATIONS_p = [
    # 'max_pool_3x3_p',
    # 'avg_pool_3x3_p',
    'conv_3x3_p',
    'conv_5x5_p',
    # 'sep_conv_3x3_p',
    # 'sep_conv_5x5_p',
    # 'dil_conv_3x3_p',
    # 'dil_conv_5x5_p',
    # 'def_conv_3x3_p',
    # 'def_conv_5x5_p',
]
ops=len(OPERATIONS_p)

OPERATIONS_n = [
    # 'max_pool_3x3_n',
    # 'avg_pool_3x3_n',
    'conv_3x3_n',
    'conv_5x5_n',
    # 'sep_conv_3x3_n',
    # 'sep_conv_5x5_n',
    # 'dil_conv_3x3_n',
    # 'dil_conv_5x5_n',
    # 'def_conv_3x3_n',
    # 'def_conv_5x5_n',

    # 'transformer',
]
mids=(3,3,2,2,1)
mids=(1,2,3,4,5)

ms=len(mids)
permutations = list({}.fromkeys(list(itertools.permutations(mids))).keys())
motifdict_c = dict(enumerate(permutations, 1))
motifdict = dict(zip(motifdict_c.values(),motifdict_c.keys()))

def convert_cell(cell_bit_string):
    # convert cell bit-string to genome
    tmp = [cell_bit_string[i:i + 2] for i in range(0, len(cell_bit_string), 2)]
    return [tmp[i:i + 2] for i in range(0, len(tmp), 2)]

def filt(sn):
    for i in range(sn.shape[0]):
        if sn[i]<1:
            sn[i]=1
        if sn[i]>120:
            sn[i]=120
    return sn
# def convert(bit_string):
#     # convert network bit-string (norm_cell + redu_cell) to genome
#     norm_gene = convert_cell(bit_string[:len(bit_string)//2])
#     redu_gene = convert_cell(bit_string[len(bit_string)//2:])
#     return [norm_gene, redu_gene]

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def sample(pops, layers, bits , ):
    sn = np.tile(mids,pops*layers).reshape(-1,ms)
    sn=shuffle_along_axis(sn,axis=1)
    sn=sn.reshape(-1,1)
    # bigmotifs=(np.random.rand(pops*layers*ms,bits-1)<0.5).astype(int).reshape(-1,bits-1)
    bigmotifs=(np.random.rand(pops*layers,bits-1)<0.5).astype(int).reshape(-1,bits-1)
    # bigmotifs=np.repeat(bigmotifs,ms,axis=0)
    sn=sn.reshape(pops*layers,-1)

    # glob=np.array([2 for i in range(pops)])[:,np.newaxis]

    genome=np.concatenate((sn, bigmotifs),axis=1).reshape(pops,-1)
    glob = (np.random.rand(pops,layers*layers)<0.5).astype(int).reshape(pops,layers,layers)
    # for i in range(layers):
    #     glob[:,i,i]=0

    genome=np.concatenate((genome, glob.reshape(pops,layers*layers)),axis=1).reshape(pops,-1)


    return genome,sn,bigmotifs,ms,glob.reshape(pops,layers*layers)

def reencode(sn,bigmotifs,pops):
    sn=sn.reshape(-1,ms)
    mnumber=np.array([motifdict[tuple(x)] for x in sn]).reshape(pops,-1)
    mgenome=np.concatenate((bigmotifs.reshape(pops,-1), mnumber),axis=1).reshape(pops,-1)

    return mgenome

def convert(mgenome, layers, bits):
    bigmotifs = mgenome[:,0:-layers].reshape(-1,bits-1)
    sn=mgenome[:,-layers:].reshape(-1)
    sn=filt(sn)
    ssn=np.array([list(motifdict_c[sn[i]]) for i in range(sn.shape[0])]).reshape(-1,1)
    result=np.concatenate((ssn,np.repeat(bigmotifs,ms,axis=0)),axis=1).reshape(-1,layers*bits*ms)

    return np.c_[result,np.ones((result.shape[0],1))*2]

def convert_single(mgenome, layers, bits):
    bigmotifs = mgenome[0:-layers].reshape(-1,bits-1)
    sn=mgenome[-layers:].reshape(-1)
    sn=filt(sn)
    ssn=np.array([list(motifdict_c[sn[i]]) for i in range(sn.shape[0])]).reshape(-1,1)
    result=list(np.concatenate((ssn,np.repeat(bigmotifs,ms,axis=0)),axis=1).reshape(layers*bits*ms))
    return np.array(result)

def c_single(mgenome, layers, bits):
    glob = mgenome[-layers*layers:]
    mgenome = mgenome[:-layers*layers]
    big=mgenome.reshape(layers,-1)
    bigmotifs = big[:,ms:]
    sn=big[:,0:ms]
    result = np.concatenate((sn.reshape(-1,1),np.repeat(bigmotifs,ms,axis=0)),axis=1).reshape(-1,).tolist()
    result.append(glob.reshape(layers,layers))
    return result
# def decode_cell(genome, norm=True):

#     cell, cell_concat = [], list(range(2, len(genome)+2))
#     for block in genome:
#         for unit in block:
#             cell.append((PRIMITIVES[unit[0]], unit[1]))
#             if unit[1] in cell_concat:
#                 cell_concat.remove(unit[1])

#     if norm:
#         return Genotype_norm(normal=cell, normal_concat=cell_concat)
#     else:
#         return Genotype_redu(reduce=cell, reduce_concat=cell_concat)


def decode(genome):
    # decodes genome to architecture
    normal_cell = genome[0]
    reduce_cell = genome[1]

    normal, normal_concat = [], list(range(2, len(normal_cell)+2))
    reduce, reduce_concat = [], list(range(2, len(reduce_cell)+2))

    for block in normal_cell:
        for unit in block:
            normal.append((PRIMITIVES[unit[0]], unit[1]))
            if unit[1] in normal_concat:
                normal_concat.remove(unit[1])

    for block in reduce_cell:
        for unit in block:
            reduce.append((PRIMITIVES[unit[0]], unit[1]))
            if unit[1] in reduce_concat:
                reduce_concat.remove(unit[1])

    return Genotype(
        normal=normal, normal_concat=normal_concat,
        reduce=reduce, reduce_concat=reduce_concat
    )

def decode_motif(layers,bits,genome):
    # decodes genome to architecture
    motif_list=[]
    motif_ids=[]

    for b in range(0,layers*bits,bits):
        motif_id='mm'+str(genome[b])

        
        motif_ids.append(genome[b])

        normalcell=eval('motifs.%s' % motif_id)
    
        newnormal=[]
        for i in range(0,len(normalcell.normal)):
            op=normalcell.normal[i]
            if 'skip' in op[0]:
                newnormal.append(op)
                continue
            elif 'back' in op[0]:
                newnormal.append((OPERATIONS_back[genome[b+1+len(normalcell.normal)-1]],op[1]))
                continue            
            elif '_n' in op[0]:
                newnormal.append((OPERATIONS_n[genome[b+1+i]],op[1]))
                continue
            elif '_p' in op[0]:
                newnormal.append((OPERATIONS_p[genome[b+1+i]],op[1]))
                continue
        m=Genotype(normal=newnormal, normal_concat=normalcell.normal_concat,)
        motif_list.append(m)


    return motif_list,motif_ids

def compare_cell(cell_string1, cell_string2):
    cell_genome1 = convert_cell(cell_string1)
    cell_genome2 = convert_cell(cell_string2)
    cell1, cell2 = cell_genome1[:], cell_genome2[:]

    for block1 in cell1:
        for block2 in cell2:
            if block1 == block2 or block1 == block2[::-1]:
                cell2.remove(block2)
                break
    if len(cell2) > 0:
        return False
    else:
        return True


def compare(string1, string2):

    if compare_cell(string1[:len(string1)//2],
                    string2[:len(string2)//2]):
        if compare_cell(string1[len(string1)//2:],
                        string2[len(string2)//2:]):
            return True

    return False


# def debug():
#     # design to debug the encoding scheme
#     seed = 0
#     np.random.seed(seed)
#     budget = 2000
#     B, n_ops, n_cell = 5, 7, 2
#     networks = []
#     design_id = 1
#     while len(networks) < budget:
#         bit_string = []
#         for c in range(n_cell):
#             for b in range(B):
#                 bit_string += [np.random.randint(n_ops),
#                                np.random.randint(b + 2),
#                                np.random.randint(n_ops),
#                                np.random.randint(b + 2)
#                                ]

#         genome = convert(bit_string)
#         # check against evaluated networks in case of duplicates
#         doTrain = True
#         for network in networks:
#             if compare(genome, network):
#                 doTrain = False
#                 break

#         if doTrain:
#             genotype = decode(genome)
#             model = Network(16, 10, 8, False, genotype)
#             model.drop_path_prob = 0.0
#             data = torch.randn(1, 3, 32, 32)
#             output, output_aux = model(torch.autograd.Variable(data))
#             networks.append(genome)
#             design_id += 1
#             print(design_id)


if __name__ == "__main__":
    # debug()
    # genome1 = [[[[3, 0], [3, 1]], [[3, 0], [3, 1]],
    #             [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
    #            [[[0, 0], [0, 1]], [[2, 2], [0, 1]],
    #             [[0, 0], [2, 2]], [[2, 2], [0, 1]]]]
    # genome2 = [[[[3, 1], [3, 0]], [[3, 1], [3, 0]],
    #             [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
    #            [[[0, 1], [0, 0]], [[2, 2], [0, 1]],
    #             [[0, 0], [2, 2]], [[2, 2], [0, 0]]]]
    #
    # print(compare(genome1, genome2))
    # print(genome1)
    # print(genome2)
    # bit_string1 = [3,1,3,0,3,1,3,0,3,1,2,0,2,0,5,2,0,0,0,1,2,2,0,1,0,0,2,2,2,2,0,1]
    # bit_string2 = [3, 0, 3, 1, 3, 0, 3, 1, 3, 1, 2, 0, 2, 0, 5, 2,
    #                0, 0, 0, 1, 2, 2, 0, 1, 0, 0, 2, 2, 2, 2, 0, 1]
    # # print(convert(bit_string1))
    # print(compare(bit_string1, bit_string2))
    # print(decode(convert(bit_string)))

    cell_bit_string = [3, 0, 3, 1, 3, 0, 3, 1, 3, 1, 2, 0, 2, 0, 5, 2]
    # print(decode_cell(convert_cell(cell_bit_string), norm=False))
    genome,sn,bigmotifs,ms=sample(195,2,20)
    mgeno=reencode(sn,bigmotifs,195)
    convert(mgeno,2,20)

