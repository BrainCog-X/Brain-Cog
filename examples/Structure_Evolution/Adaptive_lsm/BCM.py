import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

from tools.ExperimentEnvGlobalNetworkSurvival import ExperimentEnvGlobalNetworkSurvival
from tools.MazeTurnEnvVec import MazeTurnEnvVec
import torch
device = 'cuda:0'
from LSM_LIF import LSM
from tools.LSM_helper import draw_spikes,compute_rank
from tools.update_weights import stdp,bcm
from thop import profile
from thop import clever_format
matplotlib.rcParams.update({'font.size': 18})
import brewer2mpl
from cycler import cycler
bmap = brewer2mpl.get_map('Set3', 'qualitative',6)
colors=cycler('color',bmap.mpl_colors)

num = 8
n_agent = 20
steps = 500

env = MazeTurnEnvVec(n_agent, n_steps=steps)
newenv=MazeTurnEnvVec(n_agent, n_steps=steps)
data_env = ExperimentEnvGlobalNetworkSurvival(env)
newdata_env = ExperimentEnvGlobalNetworkSurvival(newenv)

p_amount=int(num*num/10)
s_amount=4

# draw_spikes(model, 0, l_s=model.sumspikes[0], r_s=model.sumspikes[1])
evolution=100
seed=0
sum_of_env = np.zeros([evolution, n_agent])
env_r=np.zeros([steps,n_agent])
model = LSM(n_offsprings=n_agent, seed=0, liquid_density=0.02, w_liquid_scale=6, w_output_scale=8,
                primary_amount=p_amount, secondary_amount=s_amount,
                height=num, width=num, )
for e in range(evolution):

    compute_rank(model)
    model.evolve(e)


old_dis = np.ones([model.n_offsprings,])*13
X = data_env.reset()
envreward = np.zeros([model.n_offsprings, ])
fit=np.zeros([n_agent])
for i in range(steps):
    model.reset()
    out = model.predict_on_batch(X + 1,i).cpu().numpy()
    X_next, envreward, fitness, infos = data_env.step(model.out.cpu().numpy())
    food_pos = data_env.env.food_pos[:, 0, :2]
    agent_pos = data_env.env.agents_pos
    dis = ((agent_pos - food_pos) ** 2).sum(1)
    reward =np.array((np.sqrt(old_dis)-np.sqrt(dis))>0,dtype=int)
    aa=np.ones_like(reward)*-1
    bb = np.ones_like(reward)*3
    cc = np.ones_like(reward)*-3
    reward=np.where(reward == 0 , aa, reward)
    reward=np.where(envreward == 1, bb, reward)
    reward = np.where(envreward == -1, cc, reward)
    for k in range(model.n_offsprings):
        old_dis[k] = dis[k]
    bcm(model,reward,input=X)

    # stdp(model.liquid_to_output_weight_matrix,input=X)
    # print("X:", X[off])
    # print("location", data_env.env.agents_pos[off])
    # print("out",out[off])

    # draw_spikes(model, id=8, inputsize=4, l_s=model.sumspikes[0], r_s=model.sumspikes[1])
    # data_env.env.consumed_count=0

    env_r[i]=reward



# np.save("./Evo_D_D.npy",env_r)
