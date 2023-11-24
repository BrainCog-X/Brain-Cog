import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft,ifft
from scipy import signal
from scipy.fft import fftshift

trail = 4
version = 2
Iraster = torch.load(f'./result/raster_{version}_delay_{trail}.pt').cpu()
time = Iraster[:, 0]
mask = (time >= 3000) & (time < 10000)
indices = torch.where(mask)
spike = Iraster[indices[0]]
plt.figure(figsize=(20, 12))
plt.scatter(spike[:, 0], spike[:, 1], s=0.1)
plt.xlabel('time [ms]', fontsize=20)
plt.ylabel('Neuron index', fontsize=20)
plt.show(dpi=600)

data = np.array(torch.load(f'./result/I_subregion_{version}_delay_{trail}.pt').cpu())
print(data.shape)

b, a = signal.butter(2, [0.002, 0.06], 'bandpass')    #配置滤波器 8 表示滤波器的阶数
data = signal.filtfilt(b, a, data)   #data为要过滤的信号

fs = 1000
time_window = 1024
# divide = torch.load('./neuron_divide.pt')
# divide_E = divide['divide_point_E']
# print(divide_E)

brain_map = ['2','5','24c','46d','7A','7B','7m','8B','8l',
             '8m','9/46d','9/46v','10','DP','F1','F2','F5','F7',
             'MT','PBr','ProM','STPc','STPi','STPr','TEO','TEpd',
             'V1','V2','V4','TH']

def region_sxx(region):
    # plt.figure()
    # plt.plot(data[region])
    # plt.show()
    plt.figure(figsize=(16, 8))

    f, t, sxx = signal.stft(data[region], fs=fs, nperseg=time_window, noverlap=time_window / 2)
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

    for col in range(1, 30):
        f, t, sxx = signal.stft(data[col], fs=fs, nperseg=time_window, noverlap=time_window / 2)
        sm = np.vstack((sm, np.max(np.abs(sxx[f_band]), axis=0)))

    cm = plt.cm.get_cmap('jet')
    plt.pcolormesh(t, brain_map, np.abs(sm), cmap=cm, shading='auto')
    #plt.pcolormesh(t, f, sxx[5:50,:],cmap=cm)
    plt.colorbar()
    plt.ylabel('Brain Regions', fontsize=10)
    plt.xlabel('Time [min]', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()

    return np.abs(sm)


region_sxx(7)

# global_sxx()

