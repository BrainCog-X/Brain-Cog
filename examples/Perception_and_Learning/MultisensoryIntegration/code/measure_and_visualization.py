#!/usr/bin/env python
#-*- coding:utf-8 -*-
__author__ = 'Yuwei Wang'
def get_concept_dataset_dic_and_initial_weights_lst(BBSR_path):
    modality_lst = ['Auditory', 'Gustatory', 'Haptic', 'Olfactory', 'Visual']
    # load_concept_dataset_df
    df_BBSR = pd.read_excel (BBSR_path, sheet_name="Sheet1", header=0, index_col=0,
                             usecols=[0, 1, 20, 26, 34, 35, 36])

    df_BBSR.rename (
        columns={'Word': 'Concept', 'Audiation_mean': 'Auditory', 'Taste': 'Gustatory',
                 'Somatic_mean': 'Haptic',
                 "Smell": "Olfactory", "Visual_mean": "Visual"}, inplace=True)
    concept_dataset_df = df_BBSR.drop_duplicates (subset="Concept")


    # get bayes weights
    var_lst = concept_dataset_df.var ().tolist ()
    c = 1 / sum ([1 / i for i in var_lst])
    bayes_weights_lst = [c / i for i in var_lst]

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

    return final_concept_dims_dic, bayes_weights_lst
def load_binarycode_dic(filename):
    # load binarycode
    binarycode_file = open (filename, 'rb')
    binarycode_dic = pickle.load (binarycode_file)
    binarycode_file.close ()
    return binarycode_dic
def load_m_dataset_concept_set_lst(m_dataset_name):
    if m_dataset_name == "McRae":
        concept_featureslst_dic = load_McRae_concept_feature_lst ()
    elif m_dataset_name == "CSLB":
        concept_featureslst_dic = load_CSLB_concept_feature_lst ()
    return list(concept_featureslst_dic.keys())
def load_McRae_concept_feature_lst():
    McRae_file = "../data/McRae_norms.xlsx"
    df_McRae = pd.read_excel (McRae_file, sheet_name="concepts_features", header=0,
                              usecols=[0, 1])

    origin_dic = df_McRae.to_dict ('index')

    ## get concept_featureslst_dic
    concept_featureslst_dic = {}
    for index in origin_dic.keys ():
        if origin_dic[index]['Concept'] not in concept_featureslst_dic.keys ():
            concept_featureslst_dic[origin_dic[index]['Concept']] = [origin_dic[index]['Feature']]
        else:
            concept_featureslst_dic[origin_dic[index]['Concept']].append (origin_dic[index]['Feature'])

    return concept_featureslst_dic
def load_CSLB_concept_feature_lst():
    CSLB_file = "../data/CSLB_norms.xlsx"
    df_CSLB = pd.read_excel (CSLB_file, header=0,
                              usecols=[0, 2, 3])

    df_CSLB.rename (
        columns={'domain': 'Domain', 'concept': 'Concept', 'feature': 'Feature'}, inplace=True)
    origin_dic = df_CSLB.to_dict ('index')

    ## get concept_featureslst_dic
    concept_featureslst_dic = {}
    for index in origin_dic.keys ():
        if origin_dic[index]['Concept'] not in concept_featureslst_dic.keys ():
            concept_featureslst_dic[origin_dic[index]['Concept']] = [origin_dic[index]['Feature']]
        else:
            concept_featureslst_dic[origin_dic[index]['Concept']].append (origin_dic[index]['Feature'])

    return concept_featureslst_dic
def get_m_dataset_concept_k_similar_concepts_dic(m_dataset_name, overlap_concept_lst, k):
    m_dataset_concept_k_similar_concepts_dic = {}
    if m_dataset_name == "McRae":
        concept_featureslst_dic = load_McRae_concept_feature_lst ()
    elif m_dataset_name == "CSLB":
        concept_featureslst_dic = load_CSLB_concept_feature_lst()

    for each_m_concept1 in overlap_concept_lst:
            similar_concepts_overnum_tuple_lst = []
            for each_m_concept2 in overlap_concept_lst:
                if each_m_concept1 != each_m_concept2:
                    overlap_feature_lst = [each_f for each_f in concept_featureslst_dic[each_m_concept1] if
                                           each_f in concept_featureslst_dic[each_m_concept2]]

                    similar_concepts_overnum_tuple_lst.append ((each_m_concept2, len (overlap_feature_lst)))
            sorted_similar_concepts_overnum_tuple_lst = sorted (similar_concepts_overnum_tuple_lst, key=lambda x: x[1],
                                                                reverse=True)
            m_dataset_concept_k_similar_concepts_dic[each_m_concept1] = [tp[0] for tp in sorted_similar_concepts_overnum_tuple_lst][:k]
    return m_dataset_concept_k_similar_concepts_dic
def get_dataset_concept_ME_dic(origin_dataset_dic):
    dataset_concept_ME_dic = {}
    for each_concept in origin_dataset_dic.keys():
        vector =  origin_dataset_dic[each_concept]
        # Modality exclusivity is a measure of the extent to which a particular property is perceived through a single
        # perceptual modality. Where each property has a vector containing mean strength ratings for all modalities,
        # modality exclusivity is calculated as the range of values divided by the sum（极差除以总和）
        ModalityExclusivity = (max(vector) - min(vector))/(0.0+sum(vector))
        # print(each_concept, ModalityExclusivity)
        dataset_concept_ME_dic[each_concept]  = ModalityExclusivity

    return dataset_concept_ME_dic
def get_dataset_concept_k_similar_concepts_ranking_dic_dic (dataset_dic, overlap_concept_lst, ifbinary):
    dataset_concept_k_similar_concepts_ranking_dic_dic = {}
    for each_concept1 in overlap_concept_lst:
        similar_concepts_similarity_tuple_lst = []
        for each_concept2 in overlap_concept_lst:
            if each_concept1 != each_concept2:
                if ifbinary:
                    similarity = get_vec_Harmming_similarity(dataset_dic[each_concept1], dataset_dic[each_concept2])
                else:
                    similarity = get_vec_cos_similarity(dataset_dic[each_concept1], dataset_dic[each_concept2])
                similar_concepts_similarity_tuple_lst.append((each_concept2, similarity))
            sorted_similar_concepts_similarity_tuple_lst = sorted (similar_concepts_similarity_tuple_lst, key=lambda x: x[1],
                                                                reverse=True)
            similar_concepts_ranking_dic = {}
            for index, tp in enumerate(sorted_similar_concepts_similarity_tuple_lst):
                similar_concepts_ranking_dic[tp[0]] = index + 1 # as ranking
        dataset_concept_k_similar_concepts_ranking_dic_dic[each_concept1] = similar_concepts_ranking_dic
    return dataset_concept_k_similar_concepts_ranking_dic_dic
def get_vec_Harmming_similarity(concept1_vecstr, concept2_vecstr):
    from scipy.spatial import distance
    HD_similarity = 1 - distance.hamming(list(concept1_vecstr), list(concept2_vecstr))
    return HD_similarity
def get_ME_kAR_corr(binarycode_type, concept_ME_dic, m_dataset, k):
    from scipy.stats import pearsonr
    if binarycode_type == "AM":
        binarycode_dic = load_binarycode_dic("../results/AM_binarycode.pickle")
    elif binarycode_type == "IM":
        binarycode_dic = load_binarycode_dic ("../results/IM_binarycode.pickle")
    else:
        print("INPUT ERROR!")


    m_dataset_concept_set = load_m_dataset_concept_set_lst (m_dataset)
    dataset_concept_set = list (concept_dims_dic.keys ())
    overlap_concept_lst = [i for i in m_dataset_concept_set if i in dataset_concept_set]


    ifbinary = True
    dataset_concept_similar_concepts_ranking_dic_dic = get_dataset_concept_k_similar_concepts_ranking_dic_dic (
        binarycode_dic, overlap_concept_lst, ifbinary)

    m_dataset_concept_k_similar_concepts_dic = get_m_dataset_concept_k_similar_concepts_dic (m_dataset,
                                                                                             overlap_concept_lst, k)

    ME_lst = []
    raking_mean_lst = []
    for each_concept in overlap_concept_lst:
        ME = concept_ME_dic[each_concept]
        k_similar_concepts_lst_in_m_dataset = m_dataset_concept_k_similar_concepts_dic[each_concept]
        ranking_list_in_dataset = [dataset_concept_similar_concepts_ranking_dic_dic[each_concept][i] for i in
                                   k_similar_concepts_lst_in_m_dataset]

        # print("ranking_list_in_dataset: ", ranking_list_in_dataset, np.mean(ranking_list_in_dataset))
        ranking_list_in_dataset = np.array (ranking_list_in_dataset)
        ME_lst.append (ME)
        raking_mean_lst.append (np.mean (ranking_list_in_dataset))

    # corr
    rho, _ = pearsonr (ME_lst, raking_mean_lst)
    print("binarycode_type m_dataset k:", binarycode_type, m_dataset, k)
    print ("correlation: ", rho)
    return ME_lst, raking_mean_lst
def visualize_results(ME_lst, raking_mean_lst, jointplot_file):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    jointplot_x = ME_lst
    jointplot_y = raking_mean_lst

    plt.figure ()
    df = pd.DataFrame ([jointplot_x, jointplot_y])
    df_final = df.T
    df_final.rename (
        columns={0: 'Modality Exclusivity', 1: 'the Average Ranking of 3 Similar Concepts'}, inplace=True)
    sns.set_style ("darkgrid")

    sns.jointplot (data=df_final, x="Modality Exclusivity", y="the Average Ranking of 3 Similar Concepts",
                   kind="reg",truncate=False, xlim=(0, 1), ylim=(0, 100))
    plt.savefig (jointplot_file, dpi=300)
    plt.close ()




if __name__ == "__main__":
    import numpy as np
    import pickle
    import pandas as pd

    BBSR_path = "../data/BBSR-5modalities.xlsx"
    all_m_dataset_lst = ["McRae", "CSLB"]
    binarycode_type_lst = ["AM", "IM"]
    k = 3

    concept_dims_dic, _ = get_concept_dataset_dic_and_initial_weights_lst (BBSR_path)
    concept_ME_dic = get_dataset_concept_ME_dic (concept_dims_dic)
    plot_info_dic = {}
    for binarycode_type in binarycode_type_lst:
        for m_dataset in all_m_dataset_lst:
            ME_lst, raking_mean_lst = get_ME_kAR_corr(binarycode_type, concept_ME_dic, m_dataset, k)
            jointplot_file = "../results/"+m_dataset + "_" + binarycode_type + "_results.png"
            visualize_results(ME_lst, raking_mean_lst, jointplot_file)









