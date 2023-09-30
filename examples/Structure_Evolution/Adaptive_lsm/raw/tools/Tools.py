import numpy as np
import matplotlib.pyplot as plt


def calc_normalized_fitness_rank_based(fitness_offspring):
    """Function that transforms raw fitness through ranking"""

    fitness = fitness_offspring.ravel()
    ranked_fitness = fitness.argsort().argsort()
    ranked_fitness = ranked_fitness.astype(np.float32) / ranked_fitness.shape[0]
    # NOTE: gives approximataly pareto shape top 20% receives 80% of weight
    ranked_fitness = ranked_fitness ** 5
    ranked_fitness /= np.sum(ranked_fitness, keepdims=True)
    fitness_offspring_normalized = ranked_fitness.reshape(fitness.shape)
    return fitness_offspring_normalized
    # plt.figure()
    # plt.plot(np.cumsum(np.sort(self.fitness_offspring_normalized.ravel())))
    # plt.show()

def get_data_path(e, exp_name, postfix):
    """Data path results and rollout are stored in"""

    # load pickle
    data_path = "./results/" + exp_name + "/"
    import os
    if not os.path.isdir("./results/"):
        os.mkdir("./results/")
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    file_name = exp_name + "_v1_" + str(e) + "_" + postfix + "_.pkl"
    if "/rollout" in postfix:
        post_path = data_path + exp_name + "_v1_" + str(e) + "_rollout"
        if not os.path.isdir(post_path):
            os.mkdir(post_path)
    return data_path + file_name

def save_fig(e, exp_name, postfix, fig_close=True):
    """Saves current plt figure to data path"""
    path = get_data_path(e, exp_name, postfix)
    png_path = path.split(".pkl")[0] + ".png"
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    if fig_close:
        plt.close()