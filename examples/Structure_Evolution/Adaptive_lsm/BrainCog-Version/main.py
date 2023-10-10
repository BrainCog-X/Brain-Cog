import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from lsmmodel import SNN
from tools.ExperimentEnvGlobalNetworkSurvival import ExperimentEnvGlobalNetworkSurvival
from tools.MazeTurnEnvVec import MazeTurnEnvVec
import torch
import brewer2mpl
from cycler import cycler
import nsganet as engine
from pymop.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.mutation.bitflip_mutation import BinaryBitflipMutation

def randbool(size, p):
    return torch.rand(*size) < p

class Evolve(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int64)
        self.xl = lb
        self.xu = ub
        self._n_evaluated = 0  # keep track of how many architectures are sampled


    def _evaluate(self, x, out, *args, **kwargs):
        
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('Network= {}'.format(arch_id))
            objs[i, 0] = np.linalg.matrix_rank(x[i])
            self._n_evaluated += 1
        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")
    
    # report generation info to files
    print("generation = {}".format(gen))
    print("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    print('Best Genome id= {}'.format(np.argmin(pop_obj[:, 0])))

if __name__ == '__main__':

    device = 'cuda:8'
    num = 8
    n_agent = 20
    steps = 500
    liquid_size=80

    env = MazeTurnEnvVec(n_agent, n_steps=steps)
    newenv=MazeTurnEnvVec(n_agent, n_steps=steps)
    data_env = ExperimentEnvGlobalNetworkSurvival(env)
    newdata_env = ExperimentEnvGlobalNetworkSurvival(newenv)



    gens=100
    seed=0
    sum_of_env = np.zeros([gens, n_agent])
    env_r=np.zeros([steps,n_agent])

    population = torch.zeros(n_agent,liquid_size,liquid_size)

    for i in range(n_agent):
        population[i] = randbool([liquid_size, liquid_size],p=0.01).to(device).float()




    kkk = Evolve(n_var=liquid_size*liquid_size, 
                    n_obj=1, n_constr=2)
    method = engine.nsganet(pop_size=n_agent,
                            sampling=RandomSampling(var_type='custom'),
                            mutation=BinaryBitflipMutation(),
                            n_offsprings=10,
                            eliminate_duplicates=True)
    kres=minimize(kkk,
                    method,
                    callback=do_every_generations,
                    termination=('n_gen', gens))


        
    # lm=evolve(population, gens)

    model = SNN(ins=4,n_agent=n_agent,device=device,liquid_size=liquid_size,lsm_tau=2,lsm_th=0.2,connectivity_matrix=randbool([liquid_size, liquid_size],p=0.01).to(device).float())
    model.to(device)
    old_dis = np.ones([n_agent,])*13

    X = data_env.reset()
    envreward = np.zeros([n_agent, ])
    fit=np.zeros([n_agent])

    for i in range(steps):
        model.reset()

        out = model(torch.from_numpy(X+1).float().to(device)).cpu().detach().numpy()

        X_next, envreward, fitness, infos = data_env.step(np.argmax(out,axis=1))

        food_pos = data_env.env.food_pos[:, 0, :2]

        agent_pos = data_env.env.agents_pos
        print(agent_pos)
        dis = ((agent_pos - food_pos) ** 2).sum(1)

        reward =np.array((np.sqrt(old_dis)-np.sqrt(dis))>0,dtype=int)

        aa=np.ones_like(reward)*-1

        bb = np.ones_like(reward)*3

        cc = np.ones_like(reward)*-3

        reward=np.where(reward == 0 , aa, reward)
        reward=np.where(envreward == 1, bb, reward)
        reward = np.where(envreward == -1, cc, reward)
        old_dis= dis
        env_r[i]=reward

