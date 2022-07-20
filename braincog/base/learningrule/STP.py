import math


class short_time():
    """
        计算短期突触可塑性的变量详见Tsodyks和Markram 1997
        :param Syn:突出可塑性结构体
        :param ISI:棘突间期
        :param Nsp:突触前棘波
    """
    def __init__(self, SizeHistOutput):
        super().__init__()   
        self.SizeHistOutput = SizeHistOutput


    def syndepr(self, Syn=None, ISI=None, Nsp=None):
        """
            短期突触可塑性计算
        """
        SizeHistOutput = self.SizeHistOutput
        qu = Syn.uprev[Nsp] * math.exp(-ISI / Syn.tc_fac)
        qR = math.exp(-ISI / Syn.tc_rec)
        u = qu + Syn.use * (1.0 - qu)
        R = Syn.Rprev[Nsp] * (1.0 - Syn.uprev[Nsp]) * qR + 1.0 - qR
        Syn.uprev[(Nsp + 1) % SizeHistOutput] = u
        Syn.Rprev[(Nsp + 1) % SizeHistOutput] = R
        return R * u

    

    def set_gsyn(self, np=None, dt=None, v=None, NoiseSyn=None):
        """
            突触电流参数计算
        """
        Isyn = 0
        gsyn_AN = 0
        gsyn_G = 0

        for j in range(np.NumSynType):
            syn = np.STList[j]
            sgate = 1.0
            if (syn.Mg_gate > 0.0):
                sgate = syn.Mg_gate / (1.0 + syn.Mg_fac * math.exp(syn.Mg_slope * (syn.Mg_half - v[0])))
            Isyn += sgate * (
                np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on)) * (
                syn.Erev - v[0])
            if (syn.Erev == 0.0):
                gsyn_AN = gsyn_AN + sgate * (
                    np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on))
            else:
                gsyn_G = gsyn_G + sgate * (
                    np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on))

        
        for j in range(NoiseSyn.NumSyn):
            syn = NoiseSyn.Syn[j].STPtr
            sgate = 1.0
            if (syn.Mg_gate > 0.0):  
                sgate = syn.Mg_gate / (1.0 + syn.Mg_fac * math.exp(syn.Mg_slope * (syn.Mg_half - v)))
            Isyn += sgate * (
                np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on)) * (
                syn.Erev - v)
            if (syn.Erev == 0.0):
                gsyn_AN = gsyn_AN + sgate * (
                    np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on))
            else:
                gsyn_G = gsyn_G + sgate * (
                    np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on))

        I_tot = Isyn + np.Iinj
        return gsyn_AN, I_tot, gsyn_G
    
    
    def IDderiv(self, np=None, v=None, dt=None, dv=None, NoiseSyn=None, flag_dv=None):
        """
         定义模型的常微分方程计算单个神经元常微分方程
         :param np:神经元参数
         :param v:当前变量
         :param dt:时间步长
        """
        Isyn = 0
        gsyn_G = 0
        gsyn_AN = 0
        for j in range(np.NumSynType):
            syn = np.STList[j]
            sgate = 1.0
            if (syn.Mg_gate > 0.0):  
                sgate = syn.Mg_gate / (1.0 + syn.Mg_fac * math.exp(syn.Mg_slope * (syn.Mg_half - v[0])))
            Isyn += sgate * (
                np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on)) * (
                syn.Erev - v[0])
            if (syn.Erev == 0.0):
                gsyn_AN = gsyn_AN + sgate * (
                    np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on))
            else:
                gsyn_G = gsyn_G + sgate * (
                    np.gfOFFsyn[j] * math.exp(-dt / syn.tc_off) - np.gfONsyn[j] * math.exp(-dt / syn.tc_on))

       
        for j in range(NoiseSyn.NumSyn):
            syn = NoiseSyn.Syn[j].STPtr
            sgate = 1.0
            if (syn.Mg_gate > 0.0):  
                sgate = syn.Mg_gate / (1.0 + syn.Mg_fac * math.exp(syn.Mg_slope * (syn.Mg_half - v[0])))
            Isyn += sgate * (
                np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on)) * (
                syn.Erev - v[0])
            if (syn.Erev == 0.0):
                gsyn_AN = gsyn_AN + sgate * (
                    np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on))
            else:
                gsyn_G = gsyn_G + sgate * (
                    np.gfOFFnoise[j] * math.exp(-dt / syn.tc_off) - np.gfONnoise[j] * math.exp(-dt / syn.tc_on))

       
        I_ex = np.gL * np.sf * math.exp((v[0] - np.Vth) / np.sf)
        
        wV = np.Iinj + Isyn - np.gL * (v[0] - np.EL) + I_ex
        
        D0 = (np.Cm / np.gL) * wV

        
        if ((
                np.Iinj + Isyn) >= np.I_ref and flag_dv == 0):  
            dv[0] = -(np.gL / np.Cm) * (v[0] - np.v_dep)
            flag_regime_osc = 0
        else:
            dv[0] = (np.Iinj - np.gL * (v[0] - np.EL) - v[1] + I_ex + Isyn) / np.Cm
            flag_regime_osc = 1

        
        dD0 = np.Cm * (math.exp((v[0] - np.Vth) / np.sf) - 1)

        
        if ((v[1] > wV - D0 / np.tcw) and (v[1] < wV + D0 / np.tcw) and v[0] <= np.Vth and (
                np.Iinj + Isyn) < np.I_ref):
            dv[1] = -(np.gL * (1 - math.exp((v[0] - np.Vth) / np.sf)) + dD0 / np.tcw) * dv[0]
        else:
            dv[1] = 0
        I_tot = Isyn + np.Iinj

        return wV, D0, gsyn_AN, gsyn_G, I_tot, dv

   

    def update(self, np=None, dt=None, NoiseSyn=None, flag_dv=None):
        """
         用二阶显式龙格-库塔法积分常微分方程
         :param np:神经元参数
         :param dt:时间步长
        """
        nvar = 2
        v = [0] * 2
        dv1 = [0] * 2
        dv2 = [0] * 2
        for i in range(nvar):
            v[i] = np.v[i]
        wV, D0, gsyn_AN, gsyn_G, I_tot, dv1 = short_time(self.SizeHistOutput).IDderiv(np, v, 0.0, dv1, NoiseSyn, flag_dv)
        for i in range(nvar):
            v[i] += dt * dv1[i]
        wV, D0, gsyn_AN, gsyn_G, I_tot, dv2 = short_time(self.SizeHistOutput).IDderiv(np, v, 0.0, dv2, NoiseSyn, flag_dv)
        for i in range(nvar):
            np.v[i] += dt / 2.0 * (dv1[i] + dv2[i])
            np.dv[i] = dt / 2.0 * (dv1[i] + dv2[i])

       
        if ((np.v[1] > wV - D0 / np.tcw) and (np.v[1] < wV + D0 / np.tcw) and np.v[0] <= np.Vth):
            np.v[1] = wV - (D0 / np.tcw)

        return np, gsyn_AN, gsyn_G, I_tot
