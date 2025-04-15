import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import torch
import matplotlib.colors as mcolors

scale = 1.2
version = 1
EEG_m = np.array(torch.load(f'./result/I_subregion_{version}_{scale}.pt').cpu())
EEG_c1 = np.load('./dataset/data_awake_1ug.npy')
EEG_c2 = np.load('./dataset/data_2ug.npy')
EEG_c3 = np.load('./dataset/data_3ug.npy')
EEG_m = np.mean(EEG_m, axis=0)

lowcut = 3.0  # 下截止频率 (Hz)
highcut = 30.0  # 上截止频率 (Hz)
fs = 1000
# 使用 Butterworth 滤波器设计带通滤波器
# butter 函数的参数依次为：滤波器阶数，频率范围（归一化），滤波器类型
b, a = signal.butter(4, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')

# 应用滤波器（使用 filtfilt 实现零相位滤波）
EEG_m = signal.filtfilt(b, a, EEG_m)
EEG_C = EEG_c3
EEG_C = signal.filtfilt(b, a, EEG_C)
t = 80
mat_all = np.zeros((t, 30))
for j in range(64):
    mat = np.zeros((t, 30))
    for i in range(t):
        f, Cxy = signal.csd(EEG_m[i*1000:(i+1)*1000], EEG_C[i][j], fs=fs, nperseg=1024)
        mat[i] = np.abs(Cxy[:30])
    mat_all += mat / np.max(mat)

plt.figure(figsize=(16,8))
norm = mcolors.LogNorm(vmin=0.001, vmax=1)
cm = plt.cm.get_cmap('jet')
plt.contourf(np.linspace(0, 8, 80) ,f[:30], mat_all.T / 64, cmap=cm, levels=200)
plt.colorbar()
plt.xlabel('time/min', fontsize=20)
plt.ylabel('Frequency/Hz', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
