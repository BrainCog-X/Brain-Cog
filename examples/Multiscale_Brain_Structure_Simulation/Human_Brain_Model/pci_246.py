import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

range_list = []

for i in range(246):
    if i < 210:
        range_list.append([i * 500, (i+1) * 500])
    else:
        range_list.append([210 * 500 + (i-210)
                           * 51, 210 * 500 + (i+1-210) * 51])

def generate_rm(Iraster):
    time_window = 40
    bm1 = np.zeros((len(range_list), int(1000/time_window)))
    bm2 = np.zeros((len(range_list), int(1000/time_window)))
    bm3 = np.zeros((len(range_list), int(1000/time_window)))
    bm4 = np.zeros((len(range_list), int(1000/time_window)))
    for i in range(len(range_list)):
        for ji, j in enumerate(range(0, 1000, time_window)):

            time = Iraster[:, 0]
            mask = (time >= j) & (time < j + time_window)
            indices = torch.where(mask)
            spike = Iraster[indices[0]]
            neuron = spike[:, 1]
            mask = (neuron >= range_list[i][0]) & (neuron < range_list[i][1])
            indices = torch.where(mask)
            spike = spike[indices[0]]
            rate = len(spike) / (time_window * (range_list[i][1] - range_list[i][0]))
            bm1[i][ji] = rate

            time = Iraster[:, 0]
            mask = (time >= j+1000) & (time < j+1000 + time_window)
            indices = torch.where(mask)
            spike = Iraster[indices[0]]
            neuron = spike[:, 1]
            mask = (neuron >= range_list[i][0]) & (neuron < range_list[i][1])
            indices = torch.where(mask)
            spike = spike[indices[0]]
            rate = len(spike) / (time_window * (range_list[i][1] - range_list[i][0]))
            bm2[i][ji] = rate

            time = Iraster[:, 0]
            mask = (time >= j+2000) & (time < j+2000 + time_window)
            indices = torch.where(mask)
            spike = Iraster[indices[0]]
            neuron = spike[:, 1]
            mask = (neuron >= range_list[i][0]) & (neuron < range_list[i][1])
            indices = torch.where(mask)
            spike = spike[indices[0]]
            rate = len(spike) / (time_window * (range_list[i][1] - range_list[i][0]))
            bm3[i][ji] = rate

            time = Iraster[:, 0]
            mask = (time >= j+3000) & (time < j+3000 + time_window)
            indices = torch.where(mask)
            spike = Iraster[indices[0]]
            neuron = spike[:, 1]
            mask = (neuron >= range_list[i][0]) & (neuron < range_list[i][1])
            indices = torch.where(mask)
            spike = spike[indices[0]]
            rate = len(spike) / (time_window * (range_list[i][1] - range_list[i][0]))
            bm4[i][ji] = rate

    return bm1, bm2, bm3, bm4

def lempel_ziv_complexity(data):
    c=1
    r=1
    q=1
    k=1
    i=1
    L1 = data.shape[0]
    L2 = data.shape[1]

    while 1:
        if q == r:
            a = i+k-1
        else:
            a=L1
        if ''.join(map(str, data[i:i+k,r-1])) in ''.join(map(str, data[0:a,q-1])):
            k=k+1
            if i+k>L1:
                r=r+1
                if r>L2:
                    break
                else:
                    i=0
                    q=r-1
                    k=1
        else:
            q = q-1
            if q<1:
                c=c+1
                i=i+k
                if i+1>L1:
                    r=r+1
                    if r>L2:
                        break
                    else:
                        i=0
                        q=r-1
                        k=1
                else:
                    q=r
                    k=1
    c = c+1
    return c

scale = 0.1
version = 4
Iraster1 = torch.load(f'./result/raster_{version}_{scale}.pt').cpu()
x=0
for Iraster in [Iraster1]:
    pcis = [[], [], [], []]
    for per in range(0, 246):
        print(per)
        Iraster_p = torch.load(f'./result/raster_{version}_{scale}_{per}.pt').cpu()
        rm1, rm2, rm3, rm4  = generate_rm(Iraster)
        rm1_p, rm2_p, rm3_p, rm4_p = generate_rm(Iraster_p)

        d = rm1_p - rm1
        bm = (np.abs(d) > 0.001).astype(int)
        c = lempel_ziv_complexity(bm)
        p1 = np.mean(bm)
        HL = - p1 * np.log2(p1+1e-12) - (1 - p1) * np.log2(1 - p1)+1e-12
        L = bm.shape[0] * bm.shape[1]
        L1 = np.log2(L) / L
        pci1 = c * L1 / HL
        print(pci1)
        pcis[0].append(pci1)

        d = rm2_p - rm2
        bm = (np.abs(d) > 0.001).astype(int)
        c = lempel_ziv_complexity(bm)
        p1 = np.mean(bm)
        HL = (-p1 * np.log2(p1+1e-12) - (1 - p1) * np.log2(1 - p1)+1e-12)
        L = bm.shape[0] * bm.shape[1]
        L1 = np.log2(L) / L
        pci2 = c * L1 / HL
        print(pci2)
        pcis[1].append(pci2)

        d = rm3_p - rm3
        bm = (np.abs(d) > 0.001).astype(int)
        c = lempel_ziv_complexity(bm)
        p1 = np.mean(bm)
        HL = - p1 * np.log2(p1+1e-12) - (1 - p1) * np.log2(1 - p1)+1e-12
        L = bm.shape[0] * bm.shape[1]
        L1 = np.log2(L) / L
        pci3 = c * L1 / HL
        print(pci3)
        pcis[2].append(pci3)

        d = rm4_p - rm4
        bm = (np.abs(d) > 0.001).astype(int)
        c = lempel_ziv_complexity(bm)
        p1 = np.mean(bm)
        HL = - p1 * np.log2(p1+1e-12) - (1 - p1) * np.log2(1 - p1)+1e-12
        L = bm.shape[0] * bm.shape[1]
        L1 = np.log2(L) / L
        pci4 = c * L1 / HL
        print(pci4)
        pcis[3].append(pci4)

    np.save(f'pci_all_{version}_246.npy', pcis)