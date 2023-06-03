import numpy as np
import random
import math
import matplotlib.pyplot as plt
import scipy.io as scio


class brain_model_91():

    def __init__(self, W):
        # Defining network model parameters
        # PC In TC TI TRN
        self.weight_matrix = W
        self.parameter = np.array([[10, 4, 10, 4, 4],  # vth     : Spiking threshold for  neurons [mV]
                  [40, 10, 200, 20, 40],    # tau_v   : Membrane capacitance for inhibitory neurons [pf];
                  [12, 10, 12,  10, 10],    # Tsig     : Variance of current in the inhibitory neurons
                  [0, 4.5, 0, 4.5, 4.5],   # beta_ad : Conductance of the adaptation variable variable of neurons
                  [0, -2, 0, -2,  -2]])   # alpha_ad: Coupling of the adaptation variable variable of  neurons
        self.V_th = np.array([10, 4, 10, 4, 4])
        self.tau_v = np.array([40, 10, 200, 20, 40])
        self.Tsig = np.array([12, 10, 12,  10, 10])
        self.beta_ad = np.array([0, 4.5, 0, 4.5, 4.5])
        self.alpha_ad = np.array([0, -2, 0, -2,  -2])

        self.tau_ad = 20
        self.tau_I = 10

        self.GammaII = 15
        self.GammaIE = -10
        self.GammaEE = 15
        self.GammaEI = 15
        self.TEmean = 0.5*self.V_th[0]  # Mean current to excitatory neurons
        self.TTCmean = 0.5*self.V_th[2]  # Mean current to TC neurons
        self.TImean = -5 * self.V_th[1]
        self.TTImean = -5 * self.V_th[3]
        self.TTRNmean = -5 * self.V_th[4]
        # Simulation parameters
        self.NR = len(W)
        self.NN = 500
        self.NType = self.NN * np.array([0.8, 0.2, 0., 0., 0.])
        self.NE = int(self.NType[0])
        self.NI = int(self.NType[1])
        self.NTC = int(self.NR * self.NType[2])
        self.NTI = int(self.NR * self.NType[3])
        self.NTRN = int(self.NR * self.NType[4])
        self.NC = self.NE + self.NI
        self.NSum = int(self.NR * self.NN + self.NTC + self.NTI + self.NTRN)
        self.Ncycle = 1

        self.dt = 0.1
        self.T = 4000

        self.Delta_T = 0.5
        self.refrac = 5 / self.dt
        self.ref = self.refrac*np.zeros((self.NN, 1)).squeeze(1)

        self.gamma_c = 0.1

        self.g_m = 1

    def simulation(self):
        Gama_c = self.g_m*self.gamma_c/(1-self.gamma_c)
        c_mE = self.tau_v[0]*self.g_m
        c_mTC = self.tau_v[2]*self.g_m
        c_mI = self.tau_v[1] * (self.g_m + Gama_c)
        c_mTI = self.tau_v[3] * (self.g_m + Gama_c)
        c_mTRN = self.tau_v[4] * (self.g_m + Gama_c)

        alpha_wE = self.alpha_ad[0] * self.g_m
        alpha_wTC = self.alpha_ad[2] * self.g_m + Gama_c
        alpha_wI = self.alpha_ad[1] * (self.g_m + Gama_c)
        alpha_wTI = self.alpha_ad[3] * (self.g_m + Gama_c)
        alpha_wTRN = self.alpha_ad[4] * (self.g_m + Gama_c)

        NEmean = self.TEmean*self.g_m
        NTCmean = self.TTCmean*self.g_m

        NImean = self.TImean * (self.g_m + Gama_c)
        NTImean = self.TTImean * (self.g_m + Gama_c)
        NTRNmean = self.TTRNmean * (self.g_m + Gama_c)

        NEsig = self.Tsig[0]*self.g_m
        NTCsig = self.Tsig[2] * self.g_m

        NIsig = self.Tsig[1] * (self.g_m + Gama_c)
        NTIsig = self.Tsig[3] * (self.g_m + Gama_c)
        NTRNsig = self.Tsig[4] * (self.g_m + Gama_c)

        Vgap = Gama_c/self.NType[1]

        for i in range(self.Ncycle):
            I_total = np.zeros((self.Ncycle,self.T))
            V_total = np.zeros((self.Ncycle, self.T))

            v = np.zeros((self.NSum))
            vt = np.zeros((self.NSum))
            c_m = np.zeros((self.NSum))
            alpha_w = np.zeros((self.NSum))
            beta_ad = np.zeros((self.NSum))
            delta = np.ones((self.NSum))
            ad = np.zeros((self.NSum))
            vv = np.zeros((self.NSum))
            vv_sumE = np.zeros((self.NR))
            vv_sumI = np.zeros((self.NR))
            Iback = np.zeros((self.NSum))
            Istimu = np.zeros((self.NSum))
            Im_sp = 0
            Igap = np.zeros((self.NSum))
            Ichem = np.zeros((self.NSum))
            Ieeg = np.zeros((self.NSum))
            Ieff = np.zeros((self.NSum))
            vm1 = np.zeros((self.NSum))

            I = np.zeros((self.T))
            V = np.zeros((self.T))
            Isubregion = np.zeros((self.NR, self.T))
            Vsubregion = np.zeros((self.NR, self.T))
            weight_matrix = self.weight_matrix
            EEG = np.zeros((self.T))

            # thalamus

            Iraster = []
            for t in range(self.T):
                # if t < 500:
                #     tau_vI = 10
                #     self.GammaII = 15
                #     self.GammaIE = -10
                # if t >= 500 and t < 1000:
                #     tau_vI = 30
                #     weight_matrix = 6 * self.weight_matrix
                #     self.GammaII = 30
                #     self.GammaIE = -20
                # if t >= 1000 and t < 1500:
                #     tau_vI = 30
                #     self.GammaII = 45
                #     self.GammaIE = -30
                # if t >= 1500 and t < 2000:
                #     tau_vI = 60
                #     self.GammaII = 45
                #     self.GammaIE = -30
                # if t >= 2000 and t < 2500:
                #     tau_vI = 30
                #     self.GammaII = 45
                #     self.GammaIE = -30
                # if t >= 2500 and t < 3000:
                #     tau_vI = 20
                #     self.GammaII = 30
                #     self.GammaIE = -20
                # if t > 3000:
                #     tau_vI = 10
                #     self.GammaII = 15
                #     self.GammaIE = -10
                #     weight_matrix = self.weight_matrix
                tau_vI = 10
                c_mI = tau_vI * (self.g_m + Gama_c)
                WII = self.GammaII * c_mI / self.NI / self.dt
                WEE = self.GammaEE * c_mE / self.NE / self.dt
                WEI = self.GammaEI * c_mI / self.NE / self.dt
                WIE = self.GammaIE * c_mE / self.NI / self.dt
                Iback = Iback + self.dt / self.tau_I * (-Iback + np.random.randn((self.NSum)))
                for n in range(self.NR):
                    vv_sumE[n] = np.sum(vv[n*self.NC:n*self.NC+self.NE])
                    vv_sumI[n] = np.sum(vv[n*self.NC+self.NE:n*self.NC+self.NE+self.NI])
                for m in range(self.NR):
                    va = weight_matrix[m]*vv_sumE
                    vc = weight_matrix[m]*vv_sumI
                   # thalamus

                    vb = np.sum(va)
                    vd = np.sum(vc)

                    if m < self.NR:  # cortical connection
                        NE_range = range(m * self.NC, m * self.NC + self.NE)
                        NI_range = range(m * self.NC + self.NE, m * self.NC + self.NE+ self.NI)
                        c_m[NE_range] = c_mE
                        c_m[NI_range] = c_mI

                        alpha_w[NE_range] = alpha_wE
                        alpha_w[NI_range] = alpha_wI

                        vt[NE_range] = self.parameter[0,0]
                        vt[NI_range] = self.parameter[0,1]

                        beta_ad[NE_range] = self.parameter[3, 0]
                        beta_ad[NI_range] = self.parameter[3, 1]

                        Ieff[NE_range] = Iback[NE_range] / np.sqrt(1/(2*(self.tau_I/self.dt)))*NEsig + NEmean
                        Ieff[NI_range] = Iback[NI_range] / np.sqrt(1 / (2 * (self.tau_I / self.dt))) * NIsig + NImean
                        Ichem[NE_range] = Ichem[NE_range] + self.dt / self.tau_I * (-Ichem[NE_range] + WEE * (vb - vv[NE_range])
                                                                                    + WIE * vd)
                        Ichem[NI_range] = Ichem[NI_range] + self.dt / self.tau_I * (-Ichem[NI_range] + WII * (vd - vv[NI_range])
                                                                                    + WEI * vb)
                        Igap[NI_range] = Vgap * (np.sum(v[NI_range]) - self.NI * v[NI_range])
                    else:
                        pass
                        # thalamus
                v = v + self.dt / c_m * (-self.g_m * v + alpha_w * ad + Ieff + Ichem + Igap)
                ad = ad + self.dt / self.tau_ad * (-ad + beta_ad * v)
                vv = (v >= vt).astype(int) * (vm1 < vt).astype(int)
                # v, ad, vv = aEIF().aEIFNode(v, self.dt, c_m, self.g_m, ad, Ieff, Ichem, Igap, self.tau_ad, beta_ad, vm1, delta, vt)
                # v, ad, vv = Adth().adthNode(v, self.dt, c_m, self.g_m, alpha_w, ad, Ieff, Ichem, Igap, self.tau_ad, beta_ad, vt, vm1)
                vm1 = v

                Isp = np.where(vv == 1)[0]
                Iraster.append(np.stack((t*np.ones((len(Isp))), Isp), axis=1))

                I[t] = np.sum(Ichem)/self.NSum
                V[t] = np.sum(v)
                EEG[t] = np.sum(Ieeg)
                # for l in range(self.NR):
                #     Isubregion[l, t] = np.sum(Ichem[l*self.NC : (l+1)*self.NC])/self.NC
                #     Vsubregion[l, t] = np.sum(v[l * self.NC: (l + 1) * self.NC]) / self.NC
            I_total[i] = I
            V_total[i] = V

            plt.figure(figsize=(20,12))
            Iraster = np.concatenate(Iraster, axis=0)
            plt.scatter(Iraster[:,0]*self.dt, Iraster[:,1], s=0.05)
            plt.xlabel('time [ms]', fontsize=20)
            plt.ylabel('Neuron index', fontsize=20)
            plt.show()

W = np.array(scio.loadmat('./w_91.mat')['W'])
simulation_model = brain_model_91(W)
simulation_model.simulation()














