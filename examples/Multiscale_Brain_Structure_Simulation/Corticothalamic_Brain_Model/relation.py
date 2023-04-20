import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fftpack import fft,ifft
from scipy import signal
from scipy.fft import fftshift
data = [np.array(scio.loadmat(f'./propofol_circle/cortical_thalamus_simulationpropofol_circle__16000ms_{i}.mat')['Isubregion'])
        for i in range(2,11)]
print(len(data))
data = np.stack(data, axis=0)
print(data.shape)
data = np.mean(data, axis=0)

fs = 1000
time_window = 256

brain_map = ['2','5','24c','46d','7A','7B','7m','8B','8l',
             '8m','9/46d','9/46v','10','DP','F1','F2','F5','F7',
             'MT','PBr','ProM','STPc','STPi','STPr','TEO','TEpd',
             'V1','V2','V4','TH']

def region_sxx(region):
    plt.figure()

    f, t, sxx = signal.stft(data[region], fs=fs, nperseg=time_window, noverlap=time_window / 2)
    print(sxx.shape)
    cm = plt.cm.get_cmap('jet')
    #plt.pcolormesh(t, f[2:10], np.abs(sxx[2:10]), cmap=cm, shading='auto')
    plt.contourf(t, f[2:10], np.abs(sxx[2:10]), cmap=cm, levels=50)
    plt.colorbar()
    plt.xlabel('time/s', fontsize=20)
    plt.ylabel('Frequency/Hz', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def global_sxx():
    plt.figure()
    global_eeg = (np.sum(data[0:29], axis=0)*400 + data[29]*145)/(11600+145)
    f, t, sxx = signal.stft(global_eeg, fs=fs, nperseg=time_window, noverlap=time_window / 2)
    print(sxx.shape)
    cm = plt.cm.get_cmap('jet')
    #plt.pcolormesh(t, f[2:10], np.abs(sxx[2:10]), cmap=cm, shading='auto')
    plt.contourf(t, f[2:10], np.abs(sxx[2:10]), cmap=cm, levels=50)
    plt.colorbar()
    plt.xlabel('time/s', fontsize=20)
    plt.ylabel('Frequency/Hz', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


def compare_sxx():

    f, t, sxx = signal.stft(data[0], fs=fs, nperseg=time_window, noverlap=time_window / 2)

    f_band = range(2, 10)

    sm = np.max(np.abs(sxx[f_band]), axis=0)

    for col in range(1, 30):
        f, t, sxx = signal.stft(data[col], fs=fs, nperseg=time_window, noverlap=time_window / 2)
        sm = np.vstack((sm, np.max(np.abs(sxx[f_band]), axis=0)))

    cm = plt.cm.get_cmap('jet')
    plt.pcolormesh(t, brain_map, np.abs(sm), cmap=cm, shading='auto')
    #plt.pcolormesh(t, f, sxx[5:50,:],cmap=cm)
    plt.colorbar()
    plt.ylabel('Brain Regions', fontsize=20)
    plt.xlabel('Time [sec]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)
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

in_degree = [0.20789,0.663865,0.2743751,0.7786819,0.1575642,0.50698,
0.245374,0.591737,0.63528,0.64359,0.682052,0.1987585,0.47401603,0.70605,
0.65438,0.3745125,0.33660976,0.47846,0.59102596,0.42496,0.3636,0.46887,
0.615334,0.51524264,0.624121,0.173915,0.952201,1.11166,0.8045154,0.1715]
out_degree = [0.3955767, 0.193632, 0.25691546, 0.6560947, 0.79555384, 0.1169355,
              0.11475188, 0.3795, 0.2240975, 0.8296078, 0.818684, 0.190211,
              0.44351, 0.0526954, 0.51960411, 0.8576482, 0.63039278, 0.560234,
              0.3918598, 0.585955, 0.103708, 0.43379488, 1.0663515, 0.1752277,
              0.44966, 0.3307821, 0.7958, 1.4476645, 1.29867264, 0.312]
region_sxx(26)
sm = compare_sxx()
level1 = range(18,32)
level2 = range(33,48)
level3 = range(49,79)
level2_r = range(80,95)
level1_r = range(96,110)

plt.figure()

y = np.hstack((sm[:, level1],sm[:, level1_r]))
y = np.mean(y, axis=1)
plt.scatter(in_degree, y, c='red', marker='o')
print(np.corrcoef(in_degree,y))
w, b = fit(in_degree, y)
x = 0.1 * np.array(range(0, 15))
plt.plot(x, w * x + b, c='red', label='level 1', linestyle='--')

y = np.hstack((sm[:, level2],sm[:, level2_r]))
y = np.mean(y, axis=1)
plt.scatter(in_degree, y, c='blue', marker='^')
print(np.corrcoef(in_degree,y))
w, b = fit(in_degree, y)
x = 0.1 * np.array(range(0, 15))
plt.plot(x, w * x + b, c='blue', label='level 2')

y = sm[:, level3]
y = np.mean(y, axis=1)
plt.scatter(in_degree, y, c='green', marker=',')
print(np.corrcoef(in_degree,y))
w, b = fit(in_degree, y)
x = 0.1 * np.array(range(0, 15))
plt.plot(x, w * x + b, c='green', label='level 3', linestyle='-.')

plt.legend(fontsize=15)
plt.xlabel('In Degree', fontsize=15)
plt.ylabel('Power', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.figure()

y = np.hstack((sm[:, level1],sm[:, level1_r]))
y = np.mean(y, axis=1)
plt.scatter(out_degree, y, c='red', marker='o')
print(np.corrcoef(out_degree,y))
w, b = fit(out_degree, y)
x = 0.1 * np.array(range(0, 15))
plt.plot(x, w * x + b, c='red', label='level 1', linestyle='--')

y = np.hstack((sm[:, level2],sm[:, level2_r]))
y = np.mean(y, axis=1)
plt.scatter(out_degree, y, c='blue', marker='^')
print(np.corrcoef(out_degree,y))
w, b = fit(out_degree, y)
x = 0.1 * np.array(range(0, 15))
plt.plot(x, w * x + b, c='blue', label='level 2')

y = sm[:, level3]
y = np.mean(y, axis=1)
plt.scatter(out_degree, y, c='green', marker=',')
print(np.corrcoef(out_degree,y))
w, b = fit(out_degree, y)
x = 0.1 * np.array(range(0, 15))
plt.plot(x, w * x + b, c='green', label='level 3', linestyle='-.')

plt.legend(fontsize=15)
plt.xlabel('Out Degree', fontsize=15)
plt.ylabel('Power', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()



