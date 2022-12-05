import xlrd
import numpy as np

import random
import math
import scipy.io as sci
from braincog.base.node.node import aEIF


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
        self.p = [[-50, -44, -45, -50, -50, -45],    % vth     : Spiking threshold for  neurons [mV]
                  [100, 100, 85, 200, 20, 40],       % tau_v   : Membrane capacitance for inhibitory neurons [pf];
                  [12, 10, 10, 12, 10, 10],          % Tsig     : Variance of current in the inhibitory neurons 
                  [0, 4.5, 4.5, 0, 4.5, 4.5],        % beta_ad : Conductance of the adaptation variable variable of neurons 
                  [0, - 2, - 2, 0, -2, -2],          % alpha_ad: Coupling of the adaptation variable variable of  neurons
                  [-110, -110, -66, -60, -60, -60]]  %vr       :rest voltage for neurons [mV]
        self.tau_ad = 20
        self.tau_I = 10
        self.GammaII = 15
        self.GammaIE = -10
        self.GammaEE = 15
        self.GammaEI = 15
        self.TEmean = 5
        self.TTCmean = 5
        self.NCR = 187
        self.NTN = 26
        self.NN = 500
        self.NType = [0.60, 0.16, 0.08, 0.1, 0.02, 0.04]
        self.Ncycle = 1
        self.dt = 1
        self.T = 200
        self.gamma_c = 0.1
        self.g_m = 1
        
    def plot(self,path=None):
    data = scio.loadmat(path)
    Iraster = data['Iraster']
    t=[]
    neuron=[]
    for i in Iraster:
        t.append(i[0])
        neuron.append(i[1])
    plt.scatter(t, neuron, c='k', marker='.')
    plt.show()

    def Mouse_model(self, sheet):
        """
        Calculation of rat brain model
        """
        w = []
        for i in range(sheet.nrows):
            w.append(sheet.row_values(i))
        for i in range(sheet.nrows):
            for j in range(sheet.ncols):
                w[i][j] = w[i][j] / 100000
        p = self.p
        tau_ad = self.tau_ad
        tau_I = self.tau_I
        GammaII = self.GammaII
        GammaIE = self.GammaIE
        GammaEE = self.GammaEE
        GammaEI = self.GammaEI
        TEmean = self.TEmean
        TTCmean = self.TTCmean
        NR = sheet.nrows
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
        NSum = int(NCR * (NE + NE + NI_BC + NI_MC) + NTN * (NTC + NTI + NTRN))
        Ncycle = self.Ncycle
        dt = self.dt
        T = self.T
        gamma_c = self.gamma_c
        TI_BCmean = -5 * p[0][1]
        TI_MCmean = -5 * p[0][2]
        TTImean = -5 * p[0][4]
        TTRNmean = -5 * p[0][5]

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

        I_total = [0.0] * T
        V_total = [0.0] * T

        V_E = [[0.0] * T] * NR
        V_I_BC = [[0.0] * T] * NR
        V_I_MC = [[0.0] * T] * NR
        V_TC = [[0.0] * T] * NR
        V_TI = [[0.0] * T] * NR
        V_TRN = [[0.0] * T] * NR

        v = [0.0] * NSum

        vt = [0.0] * NSum
        c_m = [0.0] * NSum
        alpha_w = [0.0] * NSum
        beta_ad = [0.0] * NSum
        ad = [0.0] * NSum
        vv = [0.0] * NSum
        Iback = [0.0] * NSum
        Istimu = [0.0] * NSum
        Im_sp = 0
        Igap = [0.0] * NSum
        Ichem = [0.0] * NSum
        Ieff = [0.0] * NSum
        vm1 = [0.0] * NSum
        va = [[0.0] * T] * NR
        vb = [[0.0] * T] * NR
        vc = [[0.0] * T] * NR
        vd = [[0.0] * T] * NR
        ve = [[0.0] * T] * NR
        vf = [[0.0] * T] * NR

        I = [0.0] * T

        V = [0.0] * T
        Isubregion = [[0.0] * T] * NR
        Vsubregion = [[0.0] * T] * NR

        weight_matrix = w

        Iraster = []
        for t in range(T):
            for m in range(NR):
                tau_vI = 10
                GammaII = 15
                GammaIE = -10
                c_mI_BC = tau_vI * (g_m + Gama_c)
                WII = GammaII * c_mI_BC / NI_BC / dt
                WEE = GammaEE * c_mE / NE / dt
                WEI = GammaEI * c_mI_BC / NE / dt
                WIE = GammaIE * c_mE / NI_BC / dt
                for n in range(NCR):
                    va[n][t] = weight_matrix[m][n] * \
                        (sum(vv[(n - 1) * NC:(n - 1) * NC + NE]))
                    vb[n][t] = weight_matrix[m][n] * \
                        (sum(vv[(n - 1) * NC + NE:(n - 1) * NC + NE + NI_BC]))
                    vc[n][t] = weight_matrix[m][n] * \
                        (sum(vv[(n - 1) * NC + NE + NI_BC:(n - 1) * NC + NE + NI_BC + NI_MC]))

                for n in range(NCR, NCR + NTN):
                    vd[n][t] = weight_matrix[m][n] * \
                        sum(vv[NCR * NC + (n - NCR - 1) * NT:NCR * NC + (n - NCR - 1) * NT + NTC])
                    ve[n][t] = weight_matrix[m][n] * sum(vv[NCR * NC + (
                        n - NCR - 1) * NT + NTC:NCR * NC + (n - NCR - 1) * NT + NTC + NTI])
                    vf[n][t] = weight_matrix[m][n] * sum(vv[NCR * NC + (
                        n - NCR - 1) * NT + NTC + NTI:NCR * NC + (n - NCR - 1) * NT + NTC + NTI + NTRN])

                va = np.array(va)
                vb = np.array(vb)
                vc = np.array(vc)
                vd = np.array(vd)
                ve = np.array(ve)
                vf = np.array(vf)

                v_e = np.sum(va, axis=0)[t] + np.sum(vd, axis=0)[t]
                v_i = np.sum(vb, axis=0)[t] + np.sum(vc, axis=0)[t] + np.sum(ve, axis=0)[t] + np.sum(vf, axis=0)[t]

                vg = np.sum(va, axis=0)[t]
                vh = np.sum(vb, axis=0)[t]
                vi = np.sum(vc, axis=0)[t]
                vj = np.sum(vd, axis=0)[t]
                vk = np.sum(ve, axis=0)[t]
                vl = np.sum(vf, axis=0)[t]
                for i in range(NSum):
                    Iback[i] = Iback[i] + dt / tau_I * (random.normalvariate(0, 1) - Iback[i])

                c_m = np.array(c_m)
                alpha_w = np.array(alpha_w)
                vt = np.array(vt)
                v = np.array(v)
                beta_ad = np.array(beta_ad)
                Iback = np.array(Iback)
                Ichem = np.array(Ichem)
                vv = np.array(vv)
                Ieff = np.array(Ieff)
                Igap = np.array(Igap)
                if m + 1 < NCR:
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

                    v[m * NC: m * NC + NE] = p[5][0]
                    v[m * NC + NE: m * NC + NE + NI_BC] = p[5][1]
                    v[m * NC + NE + NI_BC: m * NC + NE + NI_BC + NI_MC] = p[5][2]

                    beta_ad[m * NC: m * NC + NE] = p[3][0]
                    beta_ad[m * NC + NE: m * NC + NE + NI_BC] = p[3][1]
                    beta_ad[m * NC + NE + NI_BC: m *
                            NC + NE + NI_BC + NI_MC] = p[3][2]

                    midv1 = sum(v[m * NC + NE:m * NC + NE + NI_BC])
                    midv2 = sum(
                        v[m * NC + NE + NI_MC:m * NC + NE + NI_BC + NI_MC])

                    for k in range(m * NC, m * NC + NE):
                        Ieff[k] = Iback[k] / \
                            math.sqrt(1 / (2 * (tau_I / dt))) * float(NEsig) + float(NEmean)
                        Ichem[k] = Ichem[k] + float(dt / tau_I * (-Ichem[k] + WEE * (
                            vg + vj - vv[k]) + WIE * (vh + vi + vk + vl)))

                    for k in range(m * NC + NE, m * NC + NE + NI_BC):
                        Ieff[k] = Iback[k] / math.sqrt(1 / (2 * (tau_I / dt))) * float(
                            NI_BCsig) + float(NI_BCmean)
                        Ichem[k] = Ichem[k] + \
                            float(dt / tau_I * (-Ichem[k] + WII * (vh - vv[k]) + WEI * vg))
                        Igap[k] = Vgap * (midv1 - float(NI_BC) * v[k])

                    for k in range(
                            m * NC + NE + NI_BC,
                            m * NC + NE + NI_BC + NI_MC):
                        Ieff[k] = Iback[k] / math.sqrt(1 / (2 * (tau_I / dt))) * float(
                            NI_MCsig) + float(NI_MCmean)
                        Ichem[k] = Ichem[k] + \
                            float(dt / tau_I * (-Ichem[k] + WII * (vi - vv[k]) + WEI * vg))
                        Igap[k] = Vgap * (midv2 - NI_MC * v[k])

                else:
                    c_m[NCR * NC + (m - NCR) * NT: NCR * NC +
                        (m - NCR) * NT + NTC] = c_mTC
                    c_m[NCR * NC + (m - NCR) * NT + NTC: NCR *
                        NC + (m - NCR) * NT + NTC + NTI] = c_mTI
                    c_m[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR *
                        NC + (m - NCR) * NT + NTC + NTI + NTRN] = c_mTRN

                    alpha_w[NCR * NC + (m - NCR) * NT: NCR * NC + (m - NCR) * NT + NTC] = alpha_wTC
                    alpha_w[NCR * NC + (m - NCR) * NT + NTC: NCR * NC + (m - NCR) * NT + NTC + NTI] = alpha_wTI
                    alpha_w[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR * NC + (m - NCR) * NT + NTC + NTI + NTRN] = alpha_wTRN

                    vt[NCR * NC + (m - NCR) * NT: NCR * NC +
                       (m - NCR) * NT + NTC] = p[0][3]
                    vt[NCR * NC + (m - NCR) * NT + NTC: NCR *
                       NC + (m - NCR) * NT + NTC + NTI] = p[0][4]
                    vt[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR *
                       NC + (m - NCR) * NT + NTC + NTI + NTRN] = p[0][5]

                    v[NCR * NC + (m - NCR) * NT: NCR * NC +
                      (m - NCR) * NT + NTC] = p[5][3]
                    v[NCR * NC + (m - NCR) * NT + NTC: NCR * NC +
                      (m - NCR) * NT + NTC + NTI] = p[5][4]
                    v[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR *
                      NC + (m - NCR) * NT + NTC + NTI + NTRN] = p[5][5]

                    beta_ad[NCR * NC + (m - NCR) * NT: NCR *
                            NC + (m - NCR) * NT + NTC] = p[3][3]
                    beta_ad[NCR * NC + (m - NCR) * NT + NTC: NCR *
                            NC + (m - NCR) * NT + NTC + NTI] = p[3][4]
                    beta_ad[NCR * NC + (m - NCR) * NT + NTC + NTI: NCR * NC + (m - NCR) * NT + NTC + NTI + NTRN] = p[3][5]

                    midv1 = sum(v[NCR * NC + (m - NCR) * NT +
                                NTC:NCR * NC + (m - NCR - 1) * NT + NTC + NTI])
                    midv2 = sum(v[NCR * NC + (m - NCR) * NT + NTC + \
                                NTI:NCR * NC + (m - NCR - 1) * NT + NTC + NTI + NTRN])

                    for k in range(NCR * NC + (m - NCR) * NT,
                                   NCR * NC + (m - NCR) * NT + NTC):
                        Ieff[k] = Iback[k] / \
                            math.sqrt(1 / (2 * (tau_I / dt))) * float(NTCsig) + float(NTCmean)
                        Ichem[k] = Ichem[k] + float(
                            dt / tau_I * (-Ichem[k] + WEE * (vg - vv[k])) + WIE * (vh + vj + vk + vl))
                    for k in range(NCR * NC + (m - NCR) * NT + NTC,
                                   NCR * NC + (m - NCR) * NT + NTC + NTI):
                        Ieff[k] = Iback[k] / \
                            math.sqrt(1 / (2 * (tau_I / dt))) * float(NTIsig) + float(NTImean)
                        Ichem[k] = Ichem[k] + \
                            float(dt / tau_I * (-Ichem[k] + WII * (vk - vv[k]) + WEI * vj))
                        Igap[k] = Vgap * (midv1 - NTI * v[k])

                    for k in range(NCR * NC + (m - NCR) * NT + NTC + NTI,
                                   NCR * NC + (m - NCR) * NT + NTC + NTI + NTRN):
                        Ieff[k] = Iback[k] / \
                            math.sqrt(1 / (2 * (tau_I / dt))) * float(NTRNsig) + float(NTRNmean)
                        Ichem[k] = Ichem[k] + \
                            float(dt / tau_I * (-Ichem[k] + WII * (vl - vv[k]) + WEI * vj))
                        Igap[k] = Vgap * (midv2 - NTRN * v[k])

            vm1 = np.array(vm1)
            beta_ad = np.array(beta_ad)
            ad = np.array(ad)
            v, ad, vv = aEIF().aEIFNode(v, dt, c_m, g_m, alpha_w, ad, Ieff, Ichem, Igap, tau_ad, beta_ad, vt, vm1)
            vm1 = v

            Isp = np.nonzero(vv)
            Isp = np.array(Isp[0])
            if (len(Isp) != 0):
                Isp = Isp.reshape(len(Isp), 1)
                left = [t] * len(Isp)
                left = np.array(left)
                left = left.reshape(len(left), 1)
                mide = np.concatenate((left, Isp), axis=1)
            if(len(Isp) != 0) and (len(Iraster) != 0):
                Iraster = np.concatenate((Iraster, mide), axis=0)
                print('here')
            if (len(Iraster) == 0) and (len(Isp) != 0):
                Iraster = mide
                print('first')

            print(Iraster)

            I = Ieff + Ichem + Igap
            V[t] = sum(v) / NSum
            print(t)

        sci.savemat('./100ms-.mat', mdict={'Iraster': Iraster})
       


if __name__ == '__main__':
    workbook = xlrd.open_workbook(r'W_213.xlsx')
    sheet = workbook.sheet_by_index(5)
    test = Mouse_brain()
    test.Mouse_model(sheet)
    path='100ms-.mat'
    test.plot(path)
