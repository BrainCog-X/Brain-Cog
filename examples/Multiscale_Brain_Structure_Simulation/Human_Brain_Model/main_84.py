import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scipy.io as scio
import pandas as pd
import torch
import tqdm
device = 'cuda:1'

trail = 1
version = 1
scale = 1.2
class brain_model():

    def __init__(self, W):

        self.weight_matrix = W.to(device)
        self.distance_matrix = torch.zeros_like(self.weight_matrix, device=device)
        self.speed = 1.5
        self.decay = torch.ceil(self.distance_matrix / self.speed)
        self.t_window = int(torch.max(self.decay)) + 1
        self.V_th = torch.tensor([10., 4., 10., 4., 4.], device=device)
        self.tau_v = torch.tensor([40., 10., 30., 20., 40.], device=device)
        self.Tsig = torch.tensor([12., 10., 12., 10., 10.], device=device)
        self.beta = torch.tensor([0., 4.5, 0., 4.5, 4.5], device=device)
        self.alpha_ad = torch.tensor([0., -2., 0., -2., -2.], device=device)

        self.tau_ad = 20
        self.tau_I = 10

        # Simulation parameters
        self.NR = len(W)
        self.C = 70
        self.NN = 500
        self.NType = self.NN * torch.tensor([0.80, 0.20, 0.01, 0.004, 0.004], device=device)
        self.NE = 400
        self.NI = 100
        self.NTC = 30
        self.NTI = 15
        self.NTRN = 6
        self.NC = self.NE + self.NI
        self.NSum = int((self.C) * (self.NE + self.NI) + (self.NR - self.C) * (self.NTC + self.NTI + self.NTRN))
        self.NT = self.NTC + self.NTI + self.NTRN
        print(self.NSum)
        print(self.NC)
        print(self.NTI + self.NTC + self.NTRN)

        self.Ncycle = 1
        self.dt = 1
        self.T = 80000
        self.Delta_T = 0.5
        # self.refrac = 5 / self.dt
        # self.ref = self.refrac*torch.zeros((self.NN, 1)).squeeze(1)
        self.gamma_c = 0.1
        self.g_m = 1
        self.Gama_c = self.g_m * self.gamma_c / (1 - self.gamma_c)
        self.GammaII = 15
        self.GammaIE = -10
        self.GammaEE = 15
        self.GammaEI = 15
        self.TEmean = 0.5 * self.V_th[0]  # Mean current to excitatory neurons
        self.TTCmean = 0.5 * self.V_th[2]  # Mean current to TC neurons
        self.TImean = -5 * self.V_th[1]
        self.TTImean = -5 * self.V_th[3]
        self.TTRNmean = -5 * self.V_th[4]

        self.v = torch.zeros(self.NSum, device=device)
        self.vt = torch.zeros(self.NSum, device=device)
        self.c_m = torch.zeros(self.NSum, device=device)
        self.alpha_w = torch.zeros(self.NSum, device=device)
        self.beta_ad = torch.zeros(self.NSum, device=device)
        self.delta = torch.ones(self.NSum, device=device)
        self.ad = torch.zeros(self.NSum, device=device)
        self.vv = torch.zeros(self.NSum, device=device)
        self.Iback = torch.zeros(self.NSum, device=device)
        self.Istimu = torch.zeros(self.NSum, device=device)  # stimulate current
        self.Ieff = torch.zeros(self.NSum, device=device)
        self.Nmean = torch.zeros(self.NSum, device=device)
        self.Nsig = torch.zeros(self.NSum, device=device)
        self.Igap = torch.zeros(self.NSum, device=device)
        self.Ichem = torch.zeros(self.NSum, device=device)
        self.Ieeg = torch.zeros(self.NSum, device=device)
        self.vm1 = torch.zeros(self.NSum, device=device)
        self.reset = torch.zeros(self.NSum, device=device)

        self.E_range = []
        self.I_range = []
        self.TC_range = []
        self.TI_range = []
        self.TRN_range = []
        self.divide_point_E = []
        self.divide_point_I = []
        self.divide_point_TC = []
        self.divide_point_TI_TRN = []

        for n in range(self.NR):
            if n < self.C:
                self.divide_point_E.append(list(range(n * self.NC, n * self.NC + self.NE)))
                self.divide_point_I.append(list(range(n * self.NC + self.NE, n * self.NC + self.NE + self.NI)))
                self.E_range = self.E_range + list(range(n * self.NC, n * self.NC + self.NE))
                self.I_range = self.I_range + list(
                    range(n * self.NC + self.NE, n * self.NC + self.NE + self.NI))

            else:
                s = self.C * self.NC + (n-self.C) * self.NT
                self.divide_point_TC.append(list(range(s, s + self.NTC)))
                self.divide_point_TI_TRN.append(list(range(s + self.NTC,
                                                           s + self.NTC + self.NTI))
                                                + list(range(s + self.NTC + self.NTI,
                                                             s + self.NTC + self.NTI + self.NTRN)))

                self.TC_range = self.TC_range + list(range(s,
                                                           s + self.NTC))
                self.TI_range = self.TI_range + list(range(s + self.NTC,
                                                           s + self.NTC + self.NTI))
                self.TRN_range = self.TRN_range + list(range(s + self.NTC + self.NTI,
                                                             s + self.NTC + self.NTI + self.NTRN))
        self.divide_point_E = torch.tensor(self.divide_point_E, device=device)
        self.divide_point_I = torch.tensor(self.divide_point_I, device=device)
        self.divide_point_TC = torch.tensor(self.divide_point_TC, device=device)
        self.divide_point_TI_TRN = torch.tensor(self.divide_point_TI_TRN, device=device)
        self.divide_point_CR = torch.concat((self.divide_point_E, self.divide_point_I), dim=1)
        self.divide_point_TN = torch.concat((self.divide_point_TC, self.divide_point_TI_TRN), dim=1)
        torch.save({'divide_point_E': self.divide_point_E, 'divide_point_I': self.divide_point_I,
                    'TC_range': self.TC_range, 'TI_range': self.TI_range, 'TRN_range': self.TRN_range},
                   './neuron_divide.pt')

        self.c_m[self.E_range] = self.tau_v[0] * self.g_m + 15 * (torch.rand(size=(len(self.E_range),), device=device) - 0.5)
        # self.c_m[self.E_range] = self.tau_v[0] * self.g_m
        self.c_m[self.TC_range] = self.tau_v[2] * self.g_m
        self.c_m[self.I_range] = self.tau_v[1] * (self.g_m + self.Gama_c)
        self.c_m[self.TI_range] = self.tau_v[3] * (self.g_m + self.Gama_c)
        self.c_m[self.TRN_range] = self.tau_v[4] * (self.g_m + self.Gama_c)

        self.alpha_w[self.E_range] = self.alpha_ad[0] * self.g_m
        self.alpha_w[self.TC_range] = self.alpha_ad[2] * self.g_m + self.Gama_c
        self.alpha_w[self.I_range] = self.alpha_ad[1] * (self.g_m + self.Gama_c)
        self.alpha_w[self.TI_range] = self.alpha_ad[3] * (self.g_m + self.Gama_c)
        self.alpha_w[self.TRN_range] = self.alpha_ad[4] * (self.g_m + self.Gama_c)

        self.beta_ad[self.E_range] = self.beta[0]
        self.beta_ad[self.TC_range] = self.beta[2]
        self.beta_ad[self.I_range] = self.beta[1]
        self.beta_ad[self.TI_range] = self.beta[3]
        self.beta_ad[self.TRN_range] = self.beta[4]

        self.vt[self.E_range] = self.V_th[0]
        self.vt[self.TC_range] = self.V_th[2]
        self.vt[self.I_range] = self.V_th[1]
        self.vt[self.TI_range] = self.V_th[3]
        self.vt[self.TRN_range] = self.V_th[4]

        self.reset[self.E_range] = 1
        self.reset[self.TC_range] = 1
        self.reset[self.I_range] = 0
        self.reset[self.TI_range] = 0
        self.reset[self.TRN_range] = 0


        self.Nmean[self.E_range] = self.TEmean * self.g_m
        self.Nmean[self.TC_range] = self.TTCmean * self.g_m
        self.Nmean[self.I_range] = self.TImean * (self.g_m + self.Gama_c)
        self.Nmean[self.TI_range] = self.TTImean * (self.g_m + self.Gama_c)
        self.Nmean[self.TRN_range] = self.TTRNmean * (self.g_m + self.Gama_c)

        self.Nsig[self.E_range] = self.Tsig[0] * self.g_m
        self.Nsig[self.TC_range] = self.Tsig[2] * self.g_m
        self.Nsig[self.I_range] = self.Tsig[1] * (self.g_m + self.Gama_c)
        self.Nsig[self.TI_range] = self.Tsig[3] * (self.g_m + self.Gama_c)
        self.Nsig[self.TRN_range] = self.Tsig[4] * (self.g_m + self.Gama_c)

    def simulation(self, per):

        range_E = self.E_range + self.TC_range
        range_I = self.I_range + self.TI_range + self.TRN_range
        Vgap = self.Gama_c
        weight_matrix = self.weight_matrix
        print(123)
        for i in range(self.Ncycle):
            I_total = torch.zeros((self.Ncycle, self.T), device=device)
            V_total = torch.zeros((self.Ncycle, self.T), device=device)

            V = torch.zeros(self.T, device=device)
            I_subregion = torch.zeros((self.NR, self.T), device=device)
            I_subregion_E = torch.zeros((self.NR, self.T), device=device)
            I_subregion_I = torch.zeros((self.NR, self.T), device=device)
            Vsubregion = torch.zeros((self.NR, self.T), device=device)
            EEG = torch.zeros((self.T), device=device)

            Iraster = []
            vv_sumE = torch.zeros((self.NR, self.t_window), device=device)
            vv_sumI = torch.zeros((self.NR, self.t_window), device=device)

            phase = self.T / 8
            for t in tqdm.tqdm(range(self.T)):
                if t < phase:
                    tau_vI = 20
                    self.GammaII = 15
                    self.GammaIE = -10
                elif phase <= t < 3 * phase:
                    # break
                    tau_vI = 20 + 20 * (t - phase) / phase
                    self.GammaII = 15 + 20 * (t - phase) / phase
                    self.GammaIE = -10 - 20 * (t - phase) / phase
                elif 3 * phase <= t < 5 * phase:
                    tau_vI = 60
                    self.GammaII = 55
                    self.GammaIE = -50
                elif 5 * phase <= t < 7 * phase:
                    tau_vI = 60 - 20 * (t - 5 * phase) / phase
                    self.GammaII = 55 - 20 * (t - 5 * phase) / phase
                    self.GammaIE = -50 + 20 * (t - 5 * phase) / phase
                elif 7 * phase <= t < 8 * phase:
                    tau_vI = 20
                    self.GammaII = 15
                    self.GammaIE = -10

                self.c_m[range_I] = tau_vI * (self.g_m + self.Gama_c)
                WII = self.GammaII * torch.mean(self.c_m[self.I_range])
                WEE = self.GammaEE * torch.mean(self.c_m[self.E_range])
                WEI = self.GammaEI * torch.mean(self.c_m[self.I_range])
                WIE = self.GammaIE * torch.mean(self.c_m[self.E_range])

                self.Iback = self.Iback + self.dt / self.tau_I * (-self.Iback + torch.randn(self.NSum, device=device))
                self.Ieff = (self.Iback / math.sqrt(1 / (2 * (self.tau_I / self.dt))) * self.Nsig + self.Nmean)

                temp = vv_sumE.clone()
                vv_sumE[:, 0:self.t_window - 1] = temp[:, 1:self.t_window]

                vv_sumE[:, self.t_window - 1] = torch.cat((torch.mean(self.vv[self.divide_point_E], dim=1),
                                                           torch.mean(self.vv[self.divide_point_TC], dim=1)))

                temp = vv_sumI.clone()
                vv_sumI[:, 0:self.t_window - 1] = temp[:, 1:self.t_window]
                vv_sumI[:, self.t_window - 1] = torch.cat((torch.mean(self.vv[self.divide_point_I], dim=1),
                                                           torch.mean(
                                                               self.vv[self.divide_point_TI_TRN], dim=1)))

                v_sum = torch.cat((torch.mean(self.v[self.divide_point_I], dim=1),
                                   torch.mean(self.v[self.divide_point_TI_TRN], dim=1)))
                v_sum_CR = v_sum[:self.C].reshape(-1, 1) * \
                           torch.ones((self.C, self.NI), device=device)
                v_sum_CR = v_sum_CR.reshape(-1, 1).squeeze(1)
                v_sum_TN = v_sum[self.C:].reshape(-1, 1) * \
                           torch.ones(self.NR - self.C, self.NTI + self.NTRN, device=device)
                v_sum_TN = v_sum_TN.reshape(-1, 1).squeeze(1)
                v_sum = torch.cat((v_sum_CR, v_sum_TN))

                time_decay = torch.concat(
                    (torch.concat([torch.arange(self.NR, device=device).unsqueeze(0)] * self.NR, dim=0).unsqueeze(0),
                     self.t_window - 1 - self.decay.unsqueeze(0)), dim=0)
                time_decay = list(time_decay.long())

                v_E = torch.sum(weight_matrix * vv_sumE[time_decay], dim=1)
                v_I = torch.sum(weight_matrix * vv_sumI[time_decay], dim=1)

                v_E_CR = v_E[:self.NR - 14].reshape(-1, 1) * \
                         torch.ones((self.NR - 14, self.NC), device=device)
                v_I_CR = v_I[:self.NR - 14].reshape(-1, 1) * \
                         torch.ones((self.NR - 14, self.NC), device=device)
                v_E_CR = v_E_CR.reshape(-1, 1).squeeze(1)
                v_I_CR = v_I_CR.reshape(-1, 1).squeeze(1)

                v_E_TN = v_E[self.NR - 14:].reshape(-1, 1) * \
                         torch.ones(14, self.NTC + self.NTI + self.NTRN, device=device)
                v_I_TN = v_I[self.NR - 14:].reshape(-1, 1) * \
                         torch.ones(14, self.NTC + self.NTI + self.NTRN, device=device)
                v_E_TN = v_E_TN.reshape(-1, 1).squeeze(1)
                v_I_TN = v_I_TN.reshape(-1, 1).squeeze(1)

                v_E = torch.cat((v_E_CR, v_E_TN))
                v_I = torch.cat((v_I_CR, v_I_TN))
                self.Ichem[range_E] = self.Ichem[range_E] + self.dt / self.tau_I * \
                                      (-self.Ichem[range_E] + WEE * v_E[range_E]
                                       + WIE * v_I[range_E])

                self.Ichem[range_I] = self.Ichem[range_I] + self.dt / self.tau_I * \
                                      (-self.Ichem[range_I] + WII * v_I[range_I]
                                       + WEI * v_E[range_I])
                self.Igap[range_I] = Vgap * (
                        v_sum - self.v[range_I])

                # stimulation current
                # if (300 <= t < 700):
                #     self.Istimu[self.divide_point_E[per]] = 15 - self.Ichem[self.divide_point_E[per]]
                # elif (1300 <= t < 1700):
                #     self.Istimu[self.divide_point_E[per]] = 15 - self.Ichem[self.divide_point_E[per]]
                # elif (2300 <= t < 2700):
                #     self.Istimu[self.divide_point_E[per]] = 15 - self.Ichem[self.divide_point_E[per]]
                # elif (3300 <= t < 3700):
                #     self.Istimu[self.divide_point_E[per]] = 15 - self.Ichem[self.divide_point_E[per]]
                # else:
                #     self.Istimu[self.divide_point_E[per]] = 0

                # stimulation current
                # if (300 <= t < 700):
                #     self.Istimu[self.divide_point_TC[per]] = 15 - self.Ichem[self.divide_point_TC[per]]
                # elif (1300 <= t < 1700):
                #     self.Istimu[self.divide_point_TC[per]] = 15 - self.Ichem[self.divide_point_TC[per]]
                # elif (2300 <= t < 2700):
                #     self.Istimu[self.divide_point_TC[per]] = 15 - self.Ichem[self.divide_point_TC[per]]
                # elif (3300 <= t < 3700):
                #     self.Istimu[self.divide_point_TC[per]] = 15 - self.Ichem[self.divide_point_TC[per]]
                # else:
                #     self.Istimu[self.divide_point_TC[per]] = 0


                self.v = self.v + self.dt / self.c_m * (-self.g_m * self.v +
                                                        self.alpha_w * self.ad + self.Istimu + self.Ieff + self.Ichem + self.Igap)
                self.ad = self.ad + self.dt / self.tau_ad * (-self.ad + self.beta_ad * self.v)
                self.vv = (self.v >= self.vt).float() * (self.vm1 < self.vt).float()
                self.v = self.v * (1 - self.vv * self.reset)
                # self.v = self.v * (1 - self.vv)
                self.vm1 = self.v

                Isp = torch.where(self.vv == 1)[0]
                Iraster.append(torch.stack((t * torch.ones((len(Isp)), device=device), Isp), dim=1))

                I_CR = torch.mean(self.Ichem[self.divide_point_CR], dim=1)
                I_TN = torch.mean(self.Ichem[self.divide_point_TN], dim=1)
                I_subregion[:, t] = torch.cat((I_CR, I_TN), dim=0)

            print('over')
            torch.save(I_subregion.cpu(), f'./result/I_subregion_{version}_{scale}.pt')
            # torch.save(I_subregion.cpu(), f'./result/I_subregion_{version}_{scale}.pt')

            Iraster = torch.cat(Iraster, dim=0).cpu()

            # torch.save(Iraster, f'./result/raster_{version}_{scale}:{trail}.pt')
            torch.save(Iraster, f'./result/raster_{version}_{scale}.pt')



W = np.load('./IIT_connectivity_matrix.npy')
W = torch.from_numpy(W).float()
W = W[0:84, 0:84]
new_order = list(range(0,35)) + list(range(49,84))  + list(range(35,49))
W_new = W[new_order, :][:, new_order]
M = torch.max(W_new)
W_new = W_new / M

W_new = scale * W_new.to(device)

# Converts continuous value weights to binary weights
# for i in range(len(W_new)):
#     for j in range(len(W_new)):
#         if W_new[i][j] > 0.1 * scale:
#             W_new[i][j] = scale
#         else:
#             W_new[i][j] = 0

# The entropy of the degree distribution of the connection matrix is calculated by the histogram method
# from NA import histogram_entropy
# degree = torch.sum(W_new, dim=0)
# print(histogram_entropy(degree))
simulation_model = brain_model(W_new)
simulation_model.simulation(0)
for per in range(70): # Select the brain region to be injected with the stimulation current
    simulation_model = brain_model(W_new)
    simulation_model.simulation(per)
