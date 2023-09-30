import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import numpy as np


def plot_output(model_evaluator):
    """Live visualization of model output"""

    model = model_evaluator.model
    # plot default function
    input_dim = model.n_input
    output_dim = model.n_output
    n_steps = model_evaluator.output_per_step.shape[1]

    plt.clf()
    plt.ylim([0, 1.1])
    plt.xlim([0, n_steps])
    # highest fitness
    best_agent = np.argmax(model_evaluator.fitness_per_agent)
    #best_offspring = 0
    t = np.arange(0, n_steps)
    # output_best = model_evaluator.output_per_step[best_offspring, :, :]
    output_best = model_evaluator.output_per_step[best_agent, :]

    plt.plot(t, output_best, alpha=0.2)
    print("best_offspring", best_agent)

    for i in range(1):
        plt.plot(t, model_evaluator.input_per_step[i, :, 0], alpha=0.7, color="black") #wall
        plt.plot(t, model_evaluator.input_per_step[i, :, 1], alpha=0.7, color="gray") #road


        plt.plot(t, model_evaluator.input_per_step[i, :, 2], alpha=0.9, color="red") #poison
        plt.plot(t, model_evaluator.input_per_step[i, :, 3], alpha=0.9, color="green") #food
    sns.despine()

    plt.pause(0.005)
    plt.show()
    print("output_var", np.var(model_evaluator.output_per_step))
