import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d
plt.figure( figsize=(8,8) ) 
steps=1000
t = [i for i in range(steps)]
plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }

kk=np.load('./rank.npy')
avg = np.mean(kk, axis=0)
std = np.std(kk, axis=0)
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  
r1 = gaussian_filter1d(r1, sigma=20)
r2 = gaussian_filter1d(r2, sigma=20)
y_smoothed = gaussian_filter1d(avg, sigma=20)

color = palette(0)  
ax = plt.subplot()
ax.plot(t, y_smoothed, color=color, label="Average Fitness", linewidth=3.0)
ax.fill_between(t, r1, r2, color=color, alpha=0.2)
ax.tick_params(labelsize=18)
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

ax.legend(loc='lower right', prop=font1)
plt.xlabel('generations', fontsize=18)
plt.ylabel('SP', fontsize=18)
plt.savefig('./rank.png')
plt.show()
