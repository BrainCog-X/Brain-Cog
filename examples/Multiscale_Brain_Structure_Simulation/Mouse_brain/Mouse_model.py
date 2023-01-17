import xlrd
import numpy as np

import random
import math
import scipy.io as scio
from braincog.base.node.node import *
import matplotlib.pyplot as plt


class Mouse_brain():
    """
    Rat brain model
    :param tau_ad: Suppress the time constant of the adaptive variable [ms]
    :param tau_I: Time constant for filtering synaptic input [ms]
    :param GammaII:  I to I Connectivity
    :param GammaIE: I to E Connectivity
    :param GammaEE: E to E Connectivity
    :param GammaEI: E to I Connectivity
    :param TEmean: Average current of excitatory neurons
    """

    def __init__(self):
        self.p = [[-5, -4, -5, -5, -5, -5],  # vth     : Spiking threshold for  neurons [mV]
                  [100, 100, 85, 200, 20, 40],  # tau_v   : Membrane capacitance for inhibitory neurons [pf];
                  [12, 10, 10, 12, 10, 10],  # Tsig     : Variance of current in the inhibitory neurons
                  [0, 4.5, 4.5, 0, 4.5, 4.5],  # beta_ad : Conductance of the adaptation variable variable of neurons
                  [0, - 2, - 2, 0, -2, -2],  # alpha_ad: Coupling of the adaptation variable variable of  neurons
                  [-20, -20, -20, -20, -20, -20]]  # vr       :rest voltage for neurons [mV]
        self.tau_ad = 20
        self.tau_I = 10
        self.GammaII = 15
        self.GammaIE = -10
        self.GammaEE = 15
        self.GammaEI = 15
        self.TEmean = 5
        self.TTCmean = 5
        self.NCR = 177
        self.NTN = 36
        self.NN = 500
        self.NType = [0.60, 0.16, 0.08, 0.1, 0.02, 0.04]
        self.Ncycle = 1
        self.dt = 1
        self.T = 200
        self.gamma_c = 0.1
        self.g_m = 1

    def plot(self, path=None):
        data = scio.loadmat(path)
        Iraster = data['Iraster']
        t = []
        neuron = []
        for i in Iraster:
            t.append(i[0])
            neuron.append(i[1])
        plt.scatter(t, neuron, c='k', marker='.', s=0.1)
        plt.savefig('500mouse.jpg')

    def Mouse_model(self, w):
        """
        Calculation of rat brain model
        """
        p = self.p
        tau_ad = self.tau_ad
        tau_I = self.tau_I
        GammaII = self.GammaII
        GammaIE = self.GammaIE
        GammaEE = self.GammaEE
        GammaEI = self.GammaEI
        TEmean = self.TEmean
        TTCmean = self.TTCmean
        NR = len(w)
        NCR = self.NCR
        NTN = self.NTN
        NN = self.NN
        NType = self.NType
        for i in range(len(NType)):
            NType[i] = NType[i] * NN
        NE = int(NType[0])
        NI_BC = int(NType[1])
        NI_MC = int(NType[2])
        NTC = int(NType[3])
        NTI = int(NType[4])
        NTRN = int(NType[5])
        NC = int(NE + NI_BC + NI_MC)
        NT = int(NTC + NTI + NTRN)
        NSum = int(NCR * (NE + NI_BC + NI_MC) + NTN * (NTC + NTI + NTRN))
        Ncycle = self.Ncycle
        dt = self.dt
        T = self.T
        Delta_T = 0.5  # exponential parameter
        refrac = 5 / self.dt  # refractory period [ms]
        ref = np.zeros((NSum))  # refractory counter

        gamma_c = self.gamma_c
        TI_BCmean = -20
        TI_MCmean = -20
        TTImean = -20
        TTRNmean = -20

        g_m = self.g_m
        Gama_c = g_m * gamma_c / (1 - gamma_c)
        c_mE = p[1][0] * g_m
        c_mTC = p[1][3] * g_m
        alpha_wE = p[4][0] * g_m
        alpha_wTC = p[4][3] * g_m

        c_mI_BC = p[1][1] * (g_m + Gama_c)
        c_mI_MC = p[1][2] * (g_m + Gama_c)
        c_mTI = p[1][4] * (g_m + Gama_c)
        c_mTRN = p[1][5] * (g_m + Gama_c)
        alpha_wI_BC = p[4][1] * (g_m + Gama_c)
        alpha_wI_MC = p[4][2] * (g_m + Gama_c)

        alpha_wTI = p[4][4] * (g_m + Gama_c)
        alpha_wTRN = p[4][5] * (g_m + Gama_c)

        NEmean = TEmean * g_m
        NTCmean = TTCmean * g_m

        NI_BCmean = TI_BCmean * (g_m + Gama_c)

        NI_MCmean = TI_MCmean * (g_m + Gama_c)

        NTImean = TTImean * (g_m + Gama_c)
        NTRNmean = TTRNmean * (g_m + Gama_c)

        NEsig = p[2][0] * g_m
        NTCsig = p[2][3] * g_m

        NI_BCsig = p[2][1] * (g_m + Gama_c)

        NI_MCsig = p[2][2] * (g_m + Gama_c)

        NTIsig = p[2][4] * (g_m + Gama_c)
        NTRNsig = p[2][5] * (g_m + Gama_c)
        Vgap = Gama_c / NType[1]

        I_total = np.zeros((T))
        V_total = np.zeros((T))

        V_E = np.zeros((NR, T))
        V_I_BC = np.zeros((NR, T))
        V_I_MC = np.zeros((NR, T))
        V_TC = np.zeros((NR, T))
        V_TI = np.zeros((NR, T))
        V_TRN = np.zeros((NR, T))

        v = -20 * np.ones((NSum))

        vt = np.zeros((NSum))
        vr = np.zeros((NSum))
        vm1 = np.zeros((NSum))
        c_m = np.zeros((NSum))
        alpha_w = np.zeros((NSum))
        beta_ad = np.zeros((NSum))
        ad = np.zeros((NSum))
        vv = np.zeros((NSum))
        Iback = np.zeros((NSum))
        Istimu = np.zeros((NSum))
        Im_sp = 0
        Igap = np.zeros((NSum))
        Ichem = np.zeros((NSum))
        Ieff = np.zeros((NSum))


        va = np.zeros((NR, T))
        vb = np.zeros((NR, T))
        vc = np.zeros((NR, T))
        vd = np.zeros((NR, T))
        ve = np.zeros((NR, T))
        vf = np.zeros((NR, T))

        I = np.zeros((T))

        V = np.zeros((T))
        Isubregion = np.zeros((NR, T))
        Vsubregion = np.zeros((NR, T))

        weight_matrix = w
        for m in range(NR):
            tau_vI = 10
            GammaII = 15
            GammaIE = -10
            c_mI_BC = tau_vI * (g_m + Gama_c)
            WII = GammaII * c_mI_BC / NI_BC / dt
            WEE = GammaEE * c_mE / NE / dt
            WEI = GammaEI * c_mI_BC / NE / dt
            WIE = GammaIE * c_mE / NI_BC / dt
            if m < NCR:
                c_m[m * NC: m * NC + NE] = c_mE
                c_m[m * NC + NE: m * NC + NE + NI_BC] = c_mI_BC
                c_m[m * NC + NE + NI_BC: m * NC +
                                         NE + NI_BC + NI_MC] = c_mI_MC

                alpha_w[m * NC: m * NC + NE] = alpha_wE
                alpha_w[m * NC + NE: m * NC + NE + NI_BC] = alpha_wI_BC
                alpha_w[m * NC + NE + NI_BC: m * NC +
                                             NE + NI_BC + NI_MC] = alpha_wI_MC

                vt[m * NC: m * NC + NE] = p[0][0]
                vt[m * NC + NE: m * NC + NE + NI_BC] = p[0][1]
                vt[m * NC + NE + NI_BC: m * NC +
                                        NE + NI_BC + NI_MC] = p[0][2]

                vr[m * NC: m * NC + NE] = p[5][0]
                vr[m * NC + NE: m * NC + NE + NI_BC] = p[5][1]
                vr[m * NC + NE + NI_BC: m * NC +
                                        NE + NI_BC + NI_MC] = p[5][2]

                beta_ad[m * NC: m * NC + NE] = p[3][0]
                beta_ad[m * NC + NE: m * NC + NE + NI_BC] = p[3][1]
                beta_ad[m * NC + NE + NI_BC: m *
                                             NC + NE + NI_BC + NI_MC] = p[3][2]
            else:
                c_m[NCR * NC + (m - NCR) * NT: NCR * NC +
                                               (m - NCR) * NT + NTC] = c_mTC
                c_m[NCR * NC + (m - NCR) * NT + NTC: NCR *
                                                     NC + (m - NCR) * NT + NTC + NTI] = c_mTI
                c_m[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR *
                                                           NC + (m - NCR) * NT + NTC + NTI + NTRN] = c_mTRN

                alpha_w[NCR * NC + (m - NCR) * NT: NCR * NC + (m - NCR) * NT + NTC] = alpha_wTC
                alpha_w[NCR * NC + (m - NCR) * NT + NTC: NCR * NC + (m - NCR) * NT + NTC + NTI] = alpha_wTI
                alpha_w[
                NCR * NC + (m - NCR) * NT + NTC + NTI: NCR * NC + (m - NCR) * NT + NTC + NTI + NTRN] = alpha_wTRN

                vt[NCR * NC + (m - NCR) * NT: NCR * NC +
                                              (m - NCR) * NT + NTC] = p[0][3]
                vt[NCR * NC + (m - NCR) * NT + NTC: NCR *
                                                    NC + (m - NCR) * NT + NTC + NTI] = p[0][4]
                vt[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR *
                                                          NC + (m - NCR) * NT + NTC + NTI + NTRN] = p[0][5]

                vr[NCR * NC + (m - NCR) * NT: NCR * NC +
                                              (m - NCR) * NT + NTC] = p[5][3]
                vr[NCR * NC + (m - NCR) * NT + NTC: NCR *
                                                    NC + (m - NCR) * NT + NTC + NTI] = p[5][4]
                vr[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR *
                                                          NC + (m - NCR) * NT + NTC + NTI + NTRN] = p[5][5]

                beta_ad[NCR * NC + (m - NCR) * NT: NCR *
                                                   NC + (m - NCR) * NT + NTC] = p[3][3]
                beta_ad[NCR * NC + (m - NCR) * NT + NTC: NCR *
                                                         NC + (m - NCR) * NT + NTC + NTI] = p[3][4]
                beta_ad[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR * NC + (m - NCR) * NT + NTC + NTI + NTRN] = p[3][
                    5]
        Iraster = []
        for t in range(T):
            Iback = Iback + dt / tau_I * (np.random.randn(NSum) - Iback)
            for m in range(NR):
                for n in range(NCR):
                    va[n][t] = weight_matrix[m][n] * \
                               (np.sum(vv[n * NC:n * NC + NE]))
                    vb[n][t] = weight_matrix[m][n] * \
                               (np.sum(vv[n * NC + NE:n * NC + NE + NI_BC]))
                    vc[n][t] = weight_matrix[m][n] * \
                               (np.sum(vv[n * NC + NE + NI_BC:n * NC + NE + NI_BC + NI_MC]))


                for n in range(NCR, NCR + NTN):
                    vd[n][t] = weight_matrix[m][n] * \
                               np.sum(vv[NCR * NC + (n - NCR) * NT:NCR * NC + (n - NCR) * NT + NTC])
                    ve[n][t] = weight_matrix[m][n] * np.sum(vv[NCR * NC + (
                            n - NCR) * NT + NTC:NCR * NC + (n - NCR) * NT + NTC + NTI])
                    vf[n][t] = weight_matrix[m][n] * np.sum(vv[NCR * NC + (
                            n - NCR) * NT + NTC + NTI:NCR * NC + (n - NCR) * NT + NTC + NTI + NTRN])

                v_e = np.sum(va, axis=0)[t] + np.sum(vd, axis=0)[t]
                v_i = np.sum(vb, axis=0)[t] + np.sum(vc, axis=0)[t] + np.sum(ve, axis=0)[t] + np.sum(vf, axis=0)[t]

                vg = np.sum(va, axis=0)[t]
                vh = np.sum(vb, axis=0)[t]
                vi = np.sum(vc, axis=0)[t]
                vj = np.sum(vd, axis=0)[t]
                vk = np.sum(ve, axis=0)[t]
                vl = np.sum(vf, axis=0)[t]

                if m < NCR:
                    midv1 = np.sum(v[m * NC + NE:m * NC + NE + NI_BC])
                    midv2 = np.sum(
                        v[m * NC + NE + NI_MC:m * NC + NE + NI_BC + NI_MC])

                    k = range(m * NC, m * NC + NE)
                    Ieff[k] = Iback[k] / \
                            math.sqrt(1 / (2 * (tau_I / dt))) * NEsig + NEmean
                    Ichem[k] = Ichem[k] + dt / tau_I * (-Ichem[k] + WEE * (
                            vg + vj - vv[k]) + WIE * (vh + vi + vk + vl))

                    k = range(m * NC + NE, m * NC + NE + NI_BC)
                    Ieff[k] = Iback[k] / math.sqrt(1 / (2 * (tau_I / dt))) * NI_BCsig + NI_BCmean
                    Ichem[k] = Ichem[k] + \
                               dt / tau_I * (-Ichem[k] + WII * (vh - vv[k]) + WEI * vg)
                    Igap[k] = Vgap * (midv1 - NI_BC * v[k])

                    k = range(
                            m * NC + NE + NI_BC,
                            m * NC + NE + NI_BC + NI_MC)
                    Ieff[k] = Iback[k] / math.sqrt(1 / (2 * (tau_I / dt))) * NI_MCsig + NI_MCmean
                    Ichem[k] = Ichem[k] + \
                               dt / tau_I * (-Ichem[k] + WII * (vi - vv[k]) + WEI * vg)
                    Igap[k] = Vgap * (midv2 - NI_MC * v[k])

                else:
                    midv1 = np.sum(v[NCR * NC + (m - NCR) * NT +
                                  NTC:NCR * NC + (m - NCR) * NT + NTC + NTI])
                    midv2 = np.sum(v[NCR * NC + (m - NCR) * NT + NTC + \
                                  NTI:NCR * NC + (m - NCR) * NT + NTC + NTI + NTRN])

                    k = range(NCR * NC + (m - NCR) * NT,
                                   NCR * NC + (m - NCR) * NT + NTC)
                    Ieff[k] = Iback[k] / \
                              math.sqrt(1 / (2 * (tau_I / dt))) * NTCsig + NTCmean
                    Ichem[k] = Ichem[k] + dt / tau_I * (-Ichem[k] + WEE * (vg - vv[k])) + WIE * (vh + vj + vk + vl)
                    k = range(NCR * NC + (m - NCR) * NT + NTC,
                                NCR * NC + (m - NCR) * NT + NTC + NTI)
                    Ieff[k] = Iback[k] / \
                              math.sqrt(1 / (2 * (tau_I / dt))) * NTIsig + NTImean
                    Ichem[k] = Ichem[k] + \
                               dt / tau_I * (-Ichem[k] + WII * (vk - vv[k]) + WEI * vj)
                    Igap[k] = Vgap * (midv1 - NTI * v[k])

                    k = range(NCR * NC + (m - NCR) * NT + NTC + NTI,
                                NCR * NC + (m - NCR) * NT + NTC + NTI + NTRN)
                    Ieff[k] = Iback[k] / \
                              math.sqrt(1 / (2 * (tau_I / dt))) * NTRNsig + NTRNmean
                    Ichem[k] = Ichem[k] + \
                               dt / tau_I * (-Ichem[k] + WII * (vl - vv[k]) + WEI * vj)
                    Igap[k] = Vgap * (midv2 - NTRN * v[k])

            v, ad, vv, vm1 = adth().adthNode(v, dt, c_m, g_m, alpha_w, ad, Ieff, Ichem, Igap, tau_ad, beta_ad, vt, vm1)
            # v, ad, vv, ref = aEIF().aEIFNode(v, dt, c_m, g_m, alpha_w, ad, Ieff, Ichem, Igap,
            #      tau_ad, beta_ad, Delta_T, vt, vr, refrac, ref)

            Isp = np.nonzero(vv)
            Isp = np.array(Isp[0])
            if (len(Isp) != 0):
                Isp = Isp.reshape(len(Isp), 1)
                left = [t] * len(Isp)
                left = np.array(left)
                left = left.reshape(len(left), 1)
                mide = np.concatenate((left, Isp), axis=1)
            if (len(Isp) != 0) and (len(Iraster) != 0):
                Iraster = np.concatenate((Iraster, mide), axis=0)
                print('here')
            if (len(Iraster) == 0) and (len(Isp) != 0):
                Iraster = mide
                print('first')

            print(Iraster)

            I = Ieff + Ichem + Igap
            V[t] = np.sum(v) / NSum
            print(t)

        scio.savemat('./200ms-.mat', mdict={'Iraster': Iraster})


if __name__ == '__main__':
    W = np.array(scio.loadmat('./W_213.mat')['W'])
    test = Mouse_brain()
    test.Mouse_model(W)
    path = '200ms-.mat'
    test.plot(path)
