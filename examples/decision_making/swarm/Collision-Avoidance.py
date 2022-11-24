import torch,os
from braincog.model_zoo.rsnn import RSNN
from random import randint
import math
import random
import matplotlib
# matplotlib.use("TkAgg")
import numpy as np
import  random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#os.environ["SDL_VIDEODRIVER"] = "dummy"

#parameters
N =10
WORLD_WIDTH = 500
COLLISION_THRE =25 #60 65 70
WALL_COLLISION_LIMIT=10
VISIBLE_THRE=75  #3=75/COLLISION_THRE   #3*COLLISION_THRE

#eight velocity
vel_space=[[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,-1],[-1,1]]
vel_x_small=[[0,1],[1,0],[0,-1],[1,1],[1,-1]]
vel_x_large=[[0,1],[0,-1],[-1,0],[-1,-1],[-1,1]]
vel_y_small=[[0,1],[1,0],[-1,0],[1,1],[-1,1]]
vel_y_large=[[1,0],[0,-1],[-1,0],[1,-1],[-1,-1]]

N_action=len(vel_space)
col_robot=[i for i in range(N)]
# parameters for rl+snn
C = 50
runtime = 100  # Runtime in ms for choosing action

# parameters for snn
tau = 10  # time constant of STDP
stdpwin = 10  # STDP windows in ms
Apos = 0.925
Aneg = 0.1
vr = 0  # Reset Potential
vt = 0.1  # Judge if the neurons fire or not

tau_m = 20
Rm = 0.5
tau_e = 5
# inhibition weight between output population
s_in = np.random.rand(N_action * C, N_action * C)
for i in range(N_action):
    for j in range(C):
        for k in range(C):
            s_in[i * C + j][i * C + k] = 0

#init boids with no collision
global boids
boids = np.zeros(N, dtype=[('pos', int, 2), ('vel', int, 2),('nn',RSNN)])
list_rand=[i for i in range(16)]
rand_int=random.sample(list_rand,N)
for i in range(len(rand_int)):
    boids['pos'][i,0]=np.random.uniform(int(rand_int[i]%4)*125,(int(rand_int[i]%4)+1)*125+1,1)
    boids['pos'][i,1] = np.random.uniform(int(rand_int[i]/4) * 125, (int(rand_int[i]/4) + 1) * 125 + 1, 1)
boids['vel'] = np.random.uniform(-1, 2, (N, 2))
for i_vel in range(len(boids['vel'])):
    boids['nn'][i_vel] = RSNN(N_action*2,N_action*C).cuda()
    while(boids['vel'][i_vel][0]==0 and boids['vel'][i_vel][1]==0):
        boids['vel'][i_vel] = np.random.uniform(-1, 2, (1, 2))

#update boids parameters
do_update=np.zeros(N)
distance_pre=np.zeros((N,N))
tmp_min_robot=[i for i in range(N)]
tmp_input=[i for i in range(N)]
sum_deta_tmp=np.zeros(N)
sum_deta_new=np.zeros(N)

trace_decay = 0.8
def chooseAct(Net,input,explore):
    count_group = np.zeros(N_action)
    count_output = np.zeros(N_action * C)
    if explore==-1:
        pass
    else:
        pass
    for i_train in range(runtime):
        out, dw = Net(input[:,i_train])
        # rstdp
        Net.weight_trace *= trace_decay
        Net.weight_trace += dw[0][0]

        count_output=count_output+np.array(out)
        for i in range(N_action):
            count_group[i]=count_output[i*C:(i+1)*C].sum()
        if count_group.max()>C/2:
            action=count_group.argmax()
    return action,Net
        # if t==runtime-2 and len(np.where(self.count_group==0)[0])!=len(self.count_group):
        #     self.action=self.count_group.argmax()


def update_boids(xs, ys, xvs, yvs,frame):
    global distance_pre,col_c
    # Matrix off position difference and distance
    xdiff = np.add.outer(xs, -xs)
    ydiff = np.add.outer(ys, -ys)
    distance = np.sqrt(xdiff ** 2 + ydiff ** 2)
    # Calculate the boids that are visible to every other boid   -pi/2 to pi/2
    visible = np.zeros((N, N))
    dir = np.zeros((N, N))
    col_c = WORLD_WIDTH * np.ones((N, 4))
    dir_c = np.zeros((N, 4))
    angle_towards = np.arctan2(-ydiff, -xdiff)
    angle_vel = np.arctan2(yvs, xvs)
    for i in range(N):
        for j in range(N):
            if (xvs[i] == 1 and yvs[i] == 0) or (xvs[i] == 1 and yvs[i] == 1) or (xvs[i] == 0 and yvs[i] == 1) or (
                    xvs[i] == 0 and yvs[i] == -1) or (xvs[i] == 1 and yvs[i] == -1):
                if angle_towards[i][j] < angle_vel[i] + np.pi / 2 and angle_towards[i][j] > angle_vel[i] - np.pi / 2:
                    visible[i][j] = True
                if angle_towards[i][j] > angle_vel[i] - np.pi / 2 and angle_towards[i][j] < angle_vel[i]:
                    dir[i][j]=1#right
                if angle_towards[i][j] < angle_vel[i] + np.pi / 2 and angle_towards[i][j] >= angle_vel[i]:
                    dir[i][j] = 2#left
            if xvs[i] == -1 and yvs[i] == 1:
                if (angle_towards[i][j] > angle_vel[i] - np.pi / 2 and angle_towards[i][j] < np.pi) or (
                        angle_towards[i][j] > -np.pi and angle_towards[i][j] < angle_vel[i] - 1.5 * np.pi):
                    visible[i][j] = True
                    if angle_towards[i][j] > angle_vel[i] - np.pi / 2 and angle_towards[i][j] < angle_vel[i]:
                        dir[i][j] = 1
                    if (angle_towards[i][j] < np.pi and angle_towards[i][j] >= angle_vel[i]) or (
                        angle_towards[i][j] > -np.pi and angle_towards[i][j] < angle_vel[i] - 1.5 * np.pi):
                        dir[i][j] = 2
            if xvs[i] == -1 and yvs[i] == 0:
                if (angle_towards[i][j] > np.pi / 2 and angle_towards[i][j] < np.pi) or (
                        angle_towards[i][j] > -np.pi and angle_towards[i][j] < -np.pi / 2):
                    visible[i][j] = True
                if angle_towards[i][j] > np.pi / 2 and angle_towards[i][j] < np.pi:
                    dir[i][j] = 1
                if angle_towards[i][j] >= -np.pi and angle_towards[i][j] < -np.pi / 2:
                    dir[i][j] = 2
            if xvs[i] == -1 and yvs[i] == -1:
                if (angle_towards[i][j] > -np.pi and angle_towards[i][j] < -np.pi / 4) or (
                        angle_towards[i][j] > 0.75 * np.pi and angle_towards[i][j] < np.pi):
                    visible[i][j] = True
                if (angle_towards[i][j] > 0.75 * np.pi and angle_towards[i][j] < np.pi) or (
                        angle_towards[i][j] > -np.pi and angle_towards[i][j] < angle_vel[i]):
                    dir[i][j] = 1
                if angle_towards[i][j] >= angle_vel[i] and angle_towards[i][j] < -np.pi / 4:
                    dir[i][j] = 2
    v_tmp = np.diag(np.diag(visible))
    visible = visible - v_tmp
    # the danger of collision, considering dis=6*collision
    collision = np.clip(VISIBLE_THRE/COLLISION_THRE - distance / COLLISION_THRE, 0,VISIBLE_THRE/COLLISION_THRE) * visible  # visible and in some distance 3*collision_thre
    c_tmp = np.diag(np.diag(collision))
    collision = collision - c_tmp

    if len(np.where(yvs[np.where(ys < (VISIBLE_THRE/COLLISION_THRE)*WALL_COLLISION_LIMIT)] == -1)[0])>0:
        wall_tmp=np.where(ys < (VISIBLE_THRE/COLLISION_THRE)*WALL_COLLISION_LIMIT)[0]
        for i_wall in range(len(wall_tmp)):
            if yvs[wall_tmp[i_wall]] == -1:
                col_c[wall_tmp[i_wall], 0] = ys[wall_tmp[i_wall]]
                if xvs[wall_tmp[i_wall]] >= 0:
                    dir_c[wall_tmp[i_wall], 0] = 1
                else:
                    dir_c[wall_tmp[i_wall], 1] = 2
    if len(np.where(xvs[np.where(xs < (VISIBLE_THRE/COLLISION_THRE)*WALL_COLLISION_LIMIT)]==-1)[0])>0:
        wall_tmp = np.where(xs < (VISIBLE_THRE/COLLISION_THRE)*WALL_COLLISION_LIMIT)[0]
        for i_wall in range(len(wall_tmp)):
            if xvs[wall_tmp[i_wall]] == -1:
                col_c[wall_tmp[i_wall], 1] = xs[wall_tmp[i_wall]]
                if yvs[wall_tmp[i_wall]] >= 0:
                    dir_c[wall_tmp[i_wall], 1] = 2
                else:
                    dir_c[wall_tmp[i_wall], 1] = 1
    if len(np.where(yvs[np.where((WORLD_WIDTH - ys) < (VISIBLE_THRE/COLLISION_THRE) * WALL_COLLISION_LIMIT)] == 1)[0]) > 0:
        wall_tmp = np.where((WORLD_WIDTH - ys) < (VISIBLE_THRE/COLLISION_THRE) * WALL_COLLISION_LIMIT)[0]
        for i_wall in range(len(wall_tmp)):
            if yvs[wall_tmp[i_wall]]==1:
                col_c[wall_tmp[i_wall],2] =WORLD_WIDTH - ys[wall_tmp[i_wall]]
                if xvs[wall_tmp[i_wall]]>=0:
                    dir_c[wall_tmp[i_wall],2]=2
                else:
                    dir_c[wall_tmp[i_wall], 2] = 1
    if len(np.where(xvs[np.where((WORLD_WIDTH - xs) < (VISIBLE_THRE/COLLISION_THRE)*WALL_COLLISION_LIMIT)] ==1)[0])>0:
        wall_tmp=np.where((WORLD_WIDTH - xs) < (VISIBLE_THRE/COLLISION_THRE) * WALL_COLLISION_LIMIT)[0]
        for i_wall in range(len(wall_tmp)):
            if xvs[wall_tmp[i_wall]]==1:
                col_c[wall_tmp[i_wall],3] =WORLD_WIDTH - xs[wall_tmp[i_wall]]
                if yvs[wall_tmp[i_wall]]>=0:
                    dir_c[wall_tmp[i_wall],3]=1
                else:
                    dir_c[wall_tmp[i_wall], 3] = 2
    # print(col_c)
    col_c_tmp = np.clip(VISIBLE_THRE/COLLISION_THRE - col_c / WALL_COLLISION_LIMIT, 0, VISIBLE_THRE/COLLISION_THRE)
    deta_dis_tmp = distance - distance_pre
    deta_dis = deta_dis_tmp * collision  # <0 and small is the obstacle
    collision=np.c_[collision, col_c_tmp]
    deta_dis=np.c_[deta_dis, -col_c_tmp]
    dir=np.c_[dir,dir_c]
    # print(collision,deta_dis)
    #for every agent, choose the approaching agent as input
    for i in range(N):
        if frame>1 and do_update[i]>0:
            sum_deta_new[i] = (tmp_input[i] * collision[i][tmp_min_robot[i]]).sum()
            # print(sum_deta_new[i] ,sum_deta_tmp[i] )
            if sum_deta_new[i]  < sum_deta_tmp[i] :
                r=10*(sum_deta_tmp[i]-sum_deta_new[i])
            else:
                r=-10*(sum_deta_new[i]-sum_deta_tmp[i])
            boids['nn'][i].UpdateWeight(r)
        if frame > 0:
            do_update[i] =0
            if len(np.where(deta_dis[i] < 0)[0]) > 0:
                do_update[i] += 1
                # then get the velocity direction of objects and the distance between them as the network input
                appro_index = np.where(deta_dis[i] < 0)[0]  # the input is the approching directions and distances
                # print(appro_index)
                input = []
                for j in range(len(appro_index)):
                    if appro_index[j]<=N-1:
                        xvs_input = xvs[appro_index[j]]
                        yvs_input = yvs[appro_index[j]]
                        input.append(vel_space.index([xvs_input, yvs_input]))
                    else:
                        vel_tmp=int(appro_index[j]%N)
                        input.append(vel_tmp)
                dis_tmp=np.c_[distance,col_c]
                weight = -1 * dis_tmp[i][np.where(deta_dis[i] < 0)]
                # input=input[np.argmin(weight)]
                if weight.max() - weight.min() == 0:
                    weight = np.random.randint(1, 5, weight.shape)
                    weight[0] = 4
                else:
                    k = (4 - 1) / (weight.max() - weight.min())
                    weight = 1 + k * (weight - weight.min())
                # print(input,weight)
                I = np.zeros((N_action*2, runtime))
                for j in range(len(input)):
                    # print(appro_index,input,appro_index[j],dir[i][appro_index[j]],input[j]*dir[i][appro_index[j]])
                    I[int(input[j]+N_action*(dir[i][appro_index[j]]-1))][0:runtime] = max(I[int(input[j]+N_action*(dir[i][appro_index[j]]-1))][0], weight[j])
                if random.random()<0.7:
                    action_index,boids['nn'][i] = chooseAct(boids['nn'][i],I,-1)#exploitation
                else:
                    action_index,boids['nn'][i] = chooseAct(boids['nn'][i],I, 1)  #exploration
                xvs[i] = vel_space[action_index][0]
                yvs[i] = vel_space[action_index][1]
                tmp_min_robot[i] = np.where(deta_dis[i] < 0)[0]
                tmp_input[i] = weight
                sum_deta_tmp[i] = (tmp_input[i] * collision[i][tmp_min_robot[i]]).sum()
    xs+=xvs
    ys+=yvs
    xs=np.clip(xs,0,WORLD_WIDTH)
    ys = np.clip(ys, 0, WORLD_WIDTH)
    distance_pre = distance
    if frame>=10000:
        for i in range(N):
            for j in range(N_action*2):
                I = np.zeros((N_action * 2, runtime))
                I[j][0:runtime]=4
                a=chooseAct(boids['nn'][i],I,-1)
                # print(a)
                aaa=1


def animate(frame):
    update_boids(boids['pos'][:, 0], boids['pos'][:, 1], boids['vel'][:, 0], boids['vel'][:, 1],frame)
    scatter.set_offsets(boids['pos'])
    scatter1.set_offsets(boids['pos'])

#build background
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111)
ax1.set_title('Scatter Plot')
plt.xlim(-20,520)
plt.ylim(-20,520)
plt.grid(ls='--',c='gray')
plt.xlabel('X')
plt.ylabel('Y')
# Use a scatter plot to visualize the boids
color_list=['r','b','g','y','m','c','deeppink','tomato','gold','crimson','cornsilk','darkred','greenyellow','lightcoral','mintcream',
'rosybrown']
colors=color_list[0:N]
#colors=random.sample(color_list,N)
lines=np.zeros(N)+5
scatter = ax1.scatter(boids['pos'][:, 0], boids['pos'][:, 1],s=500,alpha=0.5,linewidths=lines)
scatter1 = ax1.scatter(boids['pos'][:, 0], boids['pos'][:, 1],s=2500,c=colors,alpha=0.5)
boids_newp=boids['pos']+boids['vel']*10
for i in range(N):
    boids_linex=np.hstack((boids['pos'][i, 0],boids_newp[i,0]))
    boids_liney=np.hstack((boids['pos'][i, 1],boids_newp[i,1]))
    #line,=plt.plot(boids_linex,boids_liney,linewidth=5)
#lines = [ax1.plot(np.hstack((boids['pos'][i, 0],boids_newp[i,0])), np.hstack((boids['pos'][i, 1],boids_newp[i,1])),linewidth=5) for i in range(N)]
animation = animation.FuncAnimation(fig, animate,interval=0.001)
plt.show()