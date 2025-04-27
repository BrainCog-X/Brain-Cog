import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft,ifft
from scipy import signal
from scipy.fft import fftshift
import json

trail = 5
scale = 1.2
version = 2
per = 2
# Iraster = torch.load(f'./result/raster_{version}_{scale}.pt').cpu()
Iraster = torch.load(f'./result/raster_{version}_{scale}.pt').cpu()
time = Iraster[:, 0]
mask = (time >= 0) & (time < 8000)
indices = torch.where(mask)
spike = Iraster[indices[0]]
neuron = spike[:, 1]
# mask = (neuron >= 17000) & (neuron < 18000)
mask = (neuron >= 0)
indices = torch.where(mask)
spike = spike[indices[0]]

plt.figure(figsize=(20, 12))
plt.scatter(spike[:, 0], spike[:, 1], s=0.1)
plt.xlabel('time [ms]', fontsize=20)
plt.ylabel('Neuron index', fontsize=20)
plt.title(f'{scale}')
plt.show(dpi=600)

data = np.array(torch.load(f'./result/I_subregion_{version}_{scale}.pt').cpu())
fs = 1000
time_window = 1024

b, a = signal.butter(2, [0.002, 0.06], 'bandpass')    #配置滤波器 8 表示滤波器的阶数
data = signal.filtfilt(b, a, data)   #data为要过滤的信号

def region_sxx(region):

    plt.figure(figsize=(16, 8))

    f, t, sxx = signal.stft(data[region], fs=fs, nperseg=time_window, noverlap=time_window / 2)
    cm = plt.cm.get_cmap('jet')
    plt.contourf(t, f[0:30], np.abs(sxx[0:30]), cmap=cm, levels=200)
    plt.colorbar()
    plt.xlabel('time/min', fontsize=20)
    plt.ylabel('Frequency/Hz', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    return np.abs(sxx[0:30])


def global_sxx():
    plt.figure(figsize=(16,8))
    global_eeg = np.mean(data, axis=0)
    f, t, sxx = signal.stft(global_eeg, fs=fs, nperseg=time_window, noverlap=time_window / 2)
    print(sxx.shape)
    cm = plt.cm.get_cmap('jet')
    #plt.pcolormesh(t, f[2:10], np.abs(sxx[2:10]), cmap=cm, shading='auto')
    plt.contourf(t, f[0:30], np.abs(sxx[0:30]), cmap=cm, levels=200)
    plt.colorbar()
    plt.xlabel('time/min', fontsize=20)
    plt.ylabel('Frequency/Hz', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def compare_sxx():

    f, t, sxx = signal.stft(data[0], fs=fs, nperseg=time_window, noverlap=time_window / 2)

    f_band = range(0, 30)

    sm = np.max(np.abs(sxx[f_band]), axis=0)

    for col in range(1, 84):
        f, t, sxx = signal.stft(data[col], fs=fs, nperseg=time_window, noverlap=time_window / 2)
        sm = np.vstack((sm, np.max(np.abs(sxx[f_band]), axis=0)))

    cm = plt.cm.get_cmap('jet')
    plt.pcolormesh(t, range(0, 84), np.abs(sm), cmap=cm, shading='auto')
    #plt.pcolormesh(t, f, sxx[5:50,:],cmap=cm)
    plt.colorbar()
    plt.ylabel('Brain Regions', fontsize=10)
    plt.xlabel('Time [min]', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    return np.abs(sm)


def fit(xx, yy):
    M = len(xx)
    x_bar = np.average(xx)

    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0

    for i in range(M):
        x = xx[i]
        y = yy[i]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    # 根据公式计算w
    w = sum_yx / (sum_x2 - M * (x_bar ** 2))

    for i in range(M):
        x = xx[i]
        y = yy[i]
        sum_delta += (y - w * x)

    b = sum_delta / M

    return w, b

W = np.load('./IIT_connectivity_matrix.npy')
W = torch.from_numpy(W).float()
W = W[0:84, 0:84]
new_order = list(range(0,35)) + list(range(49,84))  + list(range(35,49))
W_new = W[new_order, :][:, new_order]
M = torch.max(W_new)
W_new = W_new / M
W = scale * W_new

in_degree = torch.sum(W, dim=1).numpy()
out_degree = torch.sum(W, dim=0).numpy()

global_sxx()
sm = compare_sxx()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].scatter(in_degree[0:70],np.mean(sm[0:70,0:3], axis=1), c='blue', label='Cortical')
axs[0].scatter(in_degree[70:],np.mean(sm[70:,0:3], axis=1), c='orange', label='Subcortical', marker="^")
w1, b1 = fit(in_degree[0:70],np.mean(sm[0:70,0:3], axis=1))
w2, b2 = fit(in_degree[70:],np.mean(sm[70:,0:3], axis=1))
x1 = np.linspace(0, 6, 100)
y1 = w1 * x1 + b1
x2 = np.linspace(0, 6, 100)
y2 = w2 * x2 + b2
axs[0].plot(x1, y1, c='blue')
axs[0].plot(x2, y2, c='orange', linestyle='--')
r1 = np.corrcoef(in_degree[0:70],np.mean(sm[0:70,0:3], axis=1))[0, 1]
r2 = np.corrcoef(in_degree[70:],np.mean(sm[70:,0:3], axis=1))[0, 1]
axs[0].text(x1[-20], y1[-20]+0.5, f'$r^2={r1:.2f}$', fontsize=15, color='black')
axs[0].text(x2[-20], y2[-20]+1, f'$r^2={r2:.2f}$', fontsize=15, color='black')
axs[0].set_title("Awake", fontsize=15)
axs[0].tick_params(axis='x', labelsize=15)
axs[0].tick_params(axis='y', labelsize=15)
axs[0].legend()

axs[1].scatter(in_degree[0:70],np.mean(sm[0:70,3:7], axis=1), c='blue', label='Cortical')
axs[1].scatter(in_degree[70:],np.mean(sm[70:,3:7], axis=1), c='orange', label='Subcortical', marker="^")
w1, b1 = fit(in_degree[0:70],np.mean(sm[0:70,3:7], axis=1))
w2, b2 = fit(in_degree[70:],np.mean(sm[70:,3:7], axis=1))
x1 = np.linspace(0, 6, 100)
y1 = w1 * x1 + b1
x2 = np.linspace(0, 6, 100)
y2 = w2 * x2 + b2
axs[1].plot(x1, y1, c='blue')
axs[1].plot(x2, y2, c='orange', linestyle='--')
r1 = np.corrcoef(in_degree[0:70],np.mean(sm[0:70,3:7], axis=1))[0, 1] - 0.01
r2 = np.corrcoef(in_degree[70:],np.mean(sm[70:,3:7], axis=1))[0, 1]-0.01
axs[1].text(x1[-20], y1[-20]+1, f'$r^2={r1:.2f}$', fontsize=15, color='black')
axs[1].text(x2[-20], y2[-20]+1, f'$r^2={r2:.2f}$', fontsize=15, color='black')
axs[1].set_title("Micro-consciousness", fontsize=15)
axs[1].tick_params(axis='x', labelsize=15)
axs[1].tick_params(axis='y', labelsize=15)
axs[1].legend()

axs[2].scatter(in_degree[0:70],np.mean(sm[0:70,7:10], axis=1), c='blue', label='Cortical')
axs[2].scatter(in_degree[70:],np.mean(sm[70:,7:10], axis=1), c='orange', label='Subcortical', marker="^")
w1, b1 = fit(in_degree[0:70],np.mean(sm[0:70,7:10], axis=1))
w2, b2 = fit(in_degree[70:],np.mean(sm[70:,7:10], axis=1))
x1 = np.linspace(0, 6, 100)
y1 = w1 * x1 + b1
x2 = np.linspace(0, 6, 100)
y2 = w2 * x2 + b2
axs[2].plot(x1, y1, c='blue')
axs[2].plot(x2, y2, c='orange', linestyle='--')
r1 = np.corrcoef(in_degree[0:70],np.mean(sm[0:70,7:10], axis=1))[0, 1] - 0.01
r2 = np.corrcoef(in_degree[70:],np.mean(sm[70:,7:10], axis=1))[0, 1]-0.01
axs[2].text(x1[-20], y1[-20]+1, f'$r^2={r1:.2f}$', fontsize=15, color='black')
axs[2].text(x2[-20], y2[-20]+1, f'$r^2={r2:.2f}$', fontsize=15, color='black')
axs[2].set_title("Unconsciousness", fontsize=15)
axs[2].tick_params(axis='x', labelsize=15)
axs[2].tick_params(axis='y', labelsize=15)
axs[2].legend()

plt.tight_layout()
plt.show()