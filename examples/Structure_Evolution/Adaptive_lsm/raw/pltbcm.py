import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
sigm=3
# mpl.rcParams['font.size']=x
plt.style.use('seaborn-whitegrid')
plt.figure( figsize=(8,8) )
ax = plt.subplot()
palette = pyplot.get_cmap('Set1')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }
steps=500
t = [i for i in range(steps)]
########################################BCM+BCM
bcm=np.load('./10rewards.npy')
for e in range(bcm.shape[0]):
    sum1=0
    sum2=0
    best_agent_id=np.argmax(np.sum(bcm[e,:,:],axis=0))
    best_agent=bcm[e,:,best_agent_id]
    best_agent=best_agent[:steps]
    for i in range(steps): #累积
        sum2=sum2+best_agent[i]
        best_agent[i]=sum2
    bcm[e]=best_agent
avg = np.mean(bcm, axis=0)
std = np.std(bcm, axis=0)
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  
y_smoothed = gaussian_filter1d(avg, sigma=40)
r1 = gaussian_filter1d(r1, sigma=40)
r2 = gaussian_filter1d(r2, sigma=40)
color = palette(0)  
ax.plot(t, y_smoothed, color=color, label="Evolved model with DA-BCM", linewidth=3.0)
ax.fill_between(t, r1, r2, color=color, alpha=0.2)
print("Evolved model with DA-BCM")
print(avg[-1],avg[-1]-r1[-1],r2[-1]-avg[-1])
########################################unbcm
unbcm=np.load('./unevolved_with_bcm.npy')
avg = np.mean(unbcm, axis=0)
std = np.std(unbcm, axis=0)
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  
y_smoothed = gaussian_filter1d(avg, sigma=sigm)
r1 = gaussian_filter1d(r1, sigma=sigm)
r2 = gaussian_filter1d(r2, sigma=sigm)
color = palette(1)  
ax.plot(t, y_smoothed, color=color, label="Unevolved model with DA-BCM", linewidth=3.0)
ax.fill_between(t, r1, r2, color=color, alpha=0.2)
print("Unevolved model with DA-BCM+DA-BCM")
print(avg[-1],avg[-1]-r1[-1],r2[-1]-avg[-1])
########################################none+bcm
nonbcm=np.load('./none_bcm.npy')
avg = np.mean(nonbcm, axis=0)
std = np.std(nonbcm, axis=0)
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  
y_smoothed = gaussian_filter1d(avg, sigma=sigm)
r1 = gaussian_filter1d(r1, sigma=sigm)
r2 = gaussian_filter1d(r2, sigma=sigm)
color = palette(2)  
ax.plot(t, y_smoothed, color=color, label="Evolved model with NONE+DA-BCM", linewidth=3.0)
ax.fill_between(t, r1, r2, color=color, alpha=0.2)
print("Evolved model with none+DA-BCM")
print(avg[-1],avg[-1]-r1[-1],r2[-1]-avg[-1])
########################################stdp+bcm
stdpbcm=np.load('./stdp_bcm.npy')
avg = np.mean(stdpbcm, axis=0)
std = np.std(stdpbcm, axis=0)
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  
y_smoothed = gaussian_filter1d(avg, sigma=sigm)
r1 = gaussian_filter1d(r1, sigma=sigm)
r2 = gaussian_filter1d(r2, sigma=sigm)
color = palette(5)  
ax.plot(t, y_smoothed, color=color, label="Evolved model with STDP+DA-BCM", linewidth=3.0)
ax.fill_between(t, r1, r2, color=color, alpha=0.2)
print("Evolved model with STDP+DA-BCM")
print(avg[-1],avg[-1]-r1[-1],r2[-1]-avg[-1])
########################################LSTM
lstm=np.load('./lstm.npy')
avg = np.mean(lstm, axis=0)
std = np.std(lstm, axis=0)
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  
y_smoothed = gaussian_filter1d(avg, sigma=sigm)
r1 = gaussian_filter1d(r1, sigma=sigm)
r2 = gaussian_filter1d(r2, sigma=sigm)
color = palette(3)  
ax.plot(t, y_smoothed, color=color, label="LSTM", linewidth=3.0)
ax.fill_between(t, r1, r2, color=color, alpha=0.2)
print("LSTM")
print(avg[-1],avg[-1]-r1[-1],r2[-1]-avg[-1])
########################################Q-learning
ql=np.load('./ql.npy')
avg = np.mean(ql, axis=0)
std = np.std(ql, axis=0)
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  
y_smoothed = gaussian_filter1d(avg, sigma=sigm)
r1 = gaussian_filter1d(r1, sigma=sigm)
r2 = gaussian_filter1d(r2, sigma=sigm)
color = palette(6)  
ax.plot(t, y_smoothed, color=color, label="Q-learning", linewidth=3.0)
ax.fill_between(t, r1, r2, color=color, alpha=0.2)
print("Q-learning")
print(avg[-1],avg[-1]-r1[-1],r2[-1]-avg[-1])
########################################STDP
stdp=np.load('./inac.npy')
avg = np.mean(stdp, axis=0)
std = np.std(stdp, axis=0)
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  
y_smoothed = gaussian_filter1d(avg, sigma=sigm)
r1 = gaussian_filter1d(r1, sigma=sigm)
r2 = gaussian_filter1d(r2, sigma=sigm)
color = palette(4)  
ax.plot(t, y_smoothed, color=color, label="Evolved STDP", linewidth=3.0)
ax.fill_between(t, r1, r2, color=color, alpha=0.2)
print("Evolved STDP")
print(avg[-1],avg[-1]-r1[-1],r2[-1]-avg[-1])




ax.tick_params(labelsize=16)
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.legend(loc='upper left', prop=font1)
plt.xlabel('Steps', fontsize=18)
plt.ylabel('Average Reward', fontsize=18)
plt.savefig('./bcm.png')
plt.show()
