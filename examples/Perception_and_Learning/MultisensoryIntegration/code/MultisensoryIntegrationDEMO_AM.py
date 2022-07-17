#!/usr/bin/env python
#-*- coding:utf-8 -*-
__author__ = 'Yuwei Wang'
import torch
import pandas as pd
from torch import nn
from BrainCog.base.node.node import LIFNode, IzhNode
from torch.nn.parameter import Parameter

class AMNet(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 givenWeights,
                 bias=False,
                 node=LIFNode,
                 threshold=5,
                 tau=0.1):
        super().__init__()
        if node is None:
            raise TypeError

        self.fc = nn.Linear(in_features=in_features,
                            out_features=out_features, bias=bias)
        self.fc.weight = Parameter(givenWeights)
        self.node = node(threshold, tau)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        x = self.fc(x)
        x = self.node(x)
        return x


def get_concept_dataset_dic_and_AM_initial_weights_lst(BBSR_path):
    modality_lst = ['Auditory', 'Gustatory', 'Haptic', 'Olfactory', 'Visual']
    # load_concept_dataset_df
    df_BBSR = pd.read_excel (BBSR_path, sheet_name="Sheet1", header=0, index_col=0,
                             usecols=[0, 1, 20, 26, 34, 35, 36])

    df_BBSR.rename (
        columns={'Word': 'Concept', 'Audiation_mean': 'Auditory', 'Taste': 'Gustatory',
                 'Somatic_mean': 'Haptic',
                 "Smell": "Olfactory", "Visual_mean": "Visual"}, inplace=True)
    concept_dataset_df = df_BBSR.drop_duplicates (subset="Concept")


    # min-max
    z_minmax = lambda x: (x - np.min (x)) / (np.max (x) - np.min (x))
    dataset_df_minmax = concept_dataset_df[modality_lst].apply (z_minmax)
    concept_dataset_df = pd.concat ([concept_dataset_df[['Concept']], dataset_df_minmax], axis=1)

    # output
    dataset_concept_dims_dic = concept_dataset_df.to_dict ("index")
    final_concept_dims_dic = {}
    for each_key in dataset_concept_dims_dic.keys ():
        each_concept_name = dataset_concept_dims_dic[each_key].pop ('Concept')
        final_concept_dims_dic[each_concept_name] = [dataset_concept_dims_dic[each_key][each_modality] for each_modality
                                                     in modality_lst]

    method_type = "pearson" # spearman pearson kendall
    corr_dataset = df_BBSR.corr (method=method_type)
    corr_dic = corr_dataset.to_dict ()
    AM_initial_weights = []
    for each_m in modality_lst:
        tmp_lst = []
        for each_n in modality_lst:
            tmp_lst.append(corr_dic[each_m][each_n])
        AM_initial_weights.append(tmp_lst)
        
    AM_initial_weights = torch.tensor(AM_initial_weights, dtype=torch.float)# 后续需要改


    return final_concept_dims_dic, AM_initial_weights

def convert_vec_into_spike_trains(each_concept_vec):
    # generate input with Poisson-encoded spikes
    tmp = torch.tensor ([each_concept_vec * time])
    rates = tmp.view (time, -1)
    vec_spike_trains = torch.bernoulli (rates).byte ()  # concept_representation
    vec_spike_trains = torch.tensor (vec_spike_trains, dtype=torch.float)
    return vec_spike_trains

def reducing_to_binarycode(post_neuron_states_lst, tolerance):
    post_neuron_states_lst = [int (i) for i in post_neuron_states_lst]

    if len (post_neuron_states_lst) % tolerance != 0:
        placeholder = [0] * (tolerance - len (post_neuron_states_lst) % tolerance)

        post_neuron_states_lst_with_placeholder = post_neuron_states_lst + placeholder
    else:
        post_neuron_states_lst_with_placeholder = post_neuron_states_lst

    post_neuron_states_lst_with_placeholder = np.array (post_neuron_states_lst_with_placeholder).reshape (-1, tolerance)
    binarycode = ""
    for sub_arr in post_neuron_states_lst_with_placeholder:
        if 1.0 in sub_arr:
            binarycode += "1"
        else:
            binarycode += "0"
    return binarycode



if __name__ == "__main__":
    import numpy as np
    import pickle

    # Dataset Reference: Binder JR, Conant LL, Humphries CJ, Fernandino L, Simons SB, Aguilar M, Desai RH.
    # Toward a brain-based componential semantic representation. Cogn Neuropsychol. 2016 May-Jun;33(3-4):130-74.
    # doi: 10.1080/02643294.2016.1147426. Epub 2016 Jun 16. PMID: 27310469.
    BBSR_path = "../data/BBSR-5modalities.xlsx"
    AM_binarycode_file = open ( "../results/AM_binarycode.pickle", "wb")

    time = 1000
    tolerance = 2

    concept_dims_dic, AM_initial_weights = get_concept_dataset_dic_and_AM_initial_weights_lst (BBSR_path)
    AM_binarycode_dic = {}
    for each_concept in concept_dims_dic.keys():
        each_concept_vec = concept_dims_dic[each_concept]
        vec_spike_trains = convert_vec_into_spike_trains(each_concept_vec)
        AMnet = AMNet(in_features=5, out_features=5, givenWeights= AM_initial_weights, node=LIFNode, threshold=5, tau=0.1)
        post_neuron_states = AMnet(vec_spike_trains)
        post_neuron_states_lst = post_neuron_states.T.reshape(1, -1).tolist()[0]
        binarycode = reducing_to_binarycode(post_neuron_states_lst, tolerance)
        AM_binarycode_dic[each_concept] = binarycode
        print("AM", each_concept, binarycode)

    pickle.dump (AM_binarycode_dic, AM_binarycode_file)
    AM_binarycode_file.close ()





