import scipy.io as scio
import math
import random as rand
import copy
import os
import pandas as pd
from BrainCog.base.learningrule.STP import short_time


class six_layer_pfc(): 
    """
    Define global parameters
    :param SizeHistOutput: Set the peak value of the number of EPSP considered to be modified
    :param SizeHistInput: Set the number of possible spikes in the input neuron
    """
    def __init__(self):
        self.pi = 3.14159265418
        self.MaxNumSTperN = 20
        self.SizeHistOutput = 10  
        self.SizeHistInput = 1000000  
        self.NumCtrPar = 5
        self.NumVar = 2
        self.NumNeuPar = 12
        self.NumSynTypePar = 8
        self.NumSynPar = 7
        self.TRUE = 1
        self.FALSE = 0

    def mex_function(self, path=None):
        """
        Create arrays and parameters related to synaptic preservation of neuronal groups
        :param CtrPar: Store electrophysiological parameters of neurons
        :param NumViewGroups: Create arrays and parameters related to synaptic preservation of neuronal groups
        """
        data = scio.loadmat(path) 
        pi = self.pi
        MaxNumSTperN = self.MaxNumSTperN
        SizeHistOutput = self.SizeHistOutput 
        SizeHistInput = self.SizeHistInput  
        NumCtrPar = self.NumCtrPar
        NumVar = self.NumVar
        NumNeuPar = self.NumNeuPar
        NumSynTypePar = self.NumSynTypePar
        NumSynPar = self.NumSynPar
        TRUE = self.TRUE
        FALSE = self.FALSE
        CtrPar = data['CtrPar']  
        NeuPar = data['NeuPar']  
        NPList = data['NPList']  
        STypPar = data['STypPar']  
        SynPar = data['SynPar']  
        SPMtx = data['SPMtx']  
        EvtMtx = data['evtmtx']  
        EvtTimes = data['evttimes']  
        ViewList = data['ViewList']  
        InpSTtrains = data['InpSTtrains'] 
        NoiseDistr = data['NoiseDistr']  
        V0 = data['V0'] 
        UniqueNum = data['UniqueNum']  
        NeuronGroupsSaveArray = data['NeuronGroupsSaveArray']  
        SimPar = data['SimPar']
        NumViewGroups = NeuronGroupsSaveArray.shape[0]
        NumNeuronsPerGroup = NeuronGroupsSaveArray.shape[1]
        UniquePrint = UniqueNum
        Tstart = int(CtrPar[0][0])
        Tstop = int(CtrPar[0][1])
        dt0 = CtrPar[0][2]
        WriteST = CtrPar[0][4]
        t_display = 0
        stop_flag = 0
        
        NumViewGroups = NeuronGroupsSaveArray.shape[0]
        NumNeuronsPerGroup = NeuronGroupsSaveArray.shape[1]
        i = NPList.shape[1]
        j = NPList.shape[0]
        N = i * j
        k = NeuPar.shape[1]
        NPtr0 = []
        NumSpike = []
        gsyn1 = []
        gsyn2 = []
        Isyn = []
        flag_osc = []
        for i in range(N):
            NPtr0.append(Neuron())
            NPtr0[i].Cm = NeuPar[0][i]
            NPtr0[i].gL = NeuPar[1][i]
            NPtr0[i].EL = NeuPar[2][i]
            NPtr0[i].sf = NeuPar[3][i]
            NPtr0[i].Vup = NeuPar[4][i]
            NPtr0[i].tcw = NeuPar[5][i]
            NPtr0[i].a = NeuPar[6][i]
            NPtr0[i].b = NeuPar[7][i]
            NPtr0[i].Vr = NeuPar[8][i]
            NPtr0[i].Vth = NeuPar[9][i]
            NPtr0[i].I_ref = NeuPar[10][i]
            NPtr0[i].v_dep = NeuPar[11][i]
            NPtr0[i].Iinj = 0
            NPtr0[i].v[0] = V0[0][i]
            NPtr0[i].v[1] = V0[1][i]
            NPtr0[i].NumSynType = 0
            NPtr0[i].NumPreSyn = 0
            for j in range(MaxNumSTperN):
                NPtr0[i].STList.append(None)
            NumSpike.append(0)
            gsyn1.append(0)
            gsyn2.append(0)
            Isyn.append(0)
            flag_osc.append(0)

        M = InpSTtrains.shape[0]
        InpNPtr0 = []
        for i in range(M):
            InpNPtr0.append(InpNeuron())
            InpNPtr0[i].SP_ind = 0
            eom_ind = SizeHistInput
            j = 0
            while (j < eom_ind):
                if (eom_ind == SizeHistInput) and (
                        InpSTtrains[i][j + 1] == -1):
                    eom_ind = j + 1
                InpNPtr0[i].SPtrain[j] = InpSTtrains[i][j]
                j = j + 1
            for j in range(eom_ind, SizeHistInput):
                InpNPtr0[i].SPtrain[j] = -1
            InpNPtr0[i].NumSynType = 0
            InpNPtr0[i].NumPreSyn = 0

        NumSpike = []
        for i in range(N + M):
            NumSpike.append(0)

        NumSynType = STypPar.shape[1]

        SynTPtr0 = []
        for i in range(NumSynType):
            SynTPtr0.append(SynType())
            SynTPtr0[i].No = i
            SynTPtr0[i].gmax = STypPar[0][i]
            SynTPtr0[i].tc_on = STypPar[1][i]
            SynTPtr0[i].tc_off = STypPar[2][i]
            SynTPtr0[i].Erev = STypPar[3][i]
            SynTPtr0[i].Mg_gate = STypPar[4][i]
            SynTPtr0[i].Mg_fac = STypPar[5][i]
            SynTPtr0[i].Mg_slope = STypPar[6][i]
            SynTPtr0[i].Mg_half = STypPar[7][i]
            SynTPtr0[i].Gsyn = SynTPtr0[i].gmax * SynTPtr0[i].tc_on * \
                SynTPtr0[i].tc_off / (SynTPtr0[i].tc_off - SynTPtr0[i].tc_on)

        numST = SynPar.shape[1]
        SPList = SPMtx
        MaxNumSyn = SPList.shape[2]

        ConMtx0 = []
        com_c = []
        for i in range(N):
            for j in range(N + M):
                com_c.append(SynList())
            ConMtx0.append(com_c)
            com_c = []
        for i in range(N):
            for j in range(N + M):

                ConMtx0[i][j].NumSyn = 0
                while int(SPList[i][j][ConMtx0[i][j].NumSyn]) > 0:
                    ConMtx0[i][j].NumSyn = ConMtx0[i][j].NumSyn + 1
                    if (ConMtx0[i][j].NumSyn >= MaxNumSyn):
                        break

                if (ConMtx0[i][j].NumSyn > 0):
                    for a in range(ConMtx0[i][j].NumSyn):
                        ConMtx0[i][j].Syn.append(Synapse())
                else:
                    ConMtx0[i][j].Syn = []
                k = 0

                for k in range(ConMtx0[i][j].NumSyn):
                    nst = SPList[i][j][k] - 1
                   
                    if (j < N):  
                        InList = FALSE
                        kk = 0
                        while (kk < NPtr0[j].NumPreSyn):
                            if (nst == NPtr0[j].PreSynList[kk]):
                                InList = TRUE
                                break
                            kk = kk + 1
                        ConMtx0[i][j].Syn[k].PreSynIdx = kk
                        if (InList == FALSE):
                            NPtr0[j].NumPreSyn = NPtr0[j].NumPreSyn + 1
                            NPtr0[j].PreSynList = [0] * NPtr0[j].NumPreSyn
                            NPtr0[j].PreSynList[kk] = nst
                            for num in range(NPtr0[j].NumPreSyn):
                                NPtr0[j].SDf.append(SynDepr())
                            NPtr0[j].SDf[kk].use = SynPar[1][nst]
                            NPtr0[j].SDf[kk].tc_rec = SynPar[2][nst]
                            NPtr0[j].SDf[kk].tc_fac = SynPar[3][nst]
                            for k2 in range(SizeHistOutput):
                                NPtr0[j].SDf[kk].Adepr[k2] = 1.0
                            NPtr0[j].SDf[kk].uprev[0] = SynPar[1][nst]
                            NPtr0[j].SDf[kk].Rprev[0] = 1.0
                        STno = int(SynPar[0][nst] - 1)
                        ConMtx0[i][j].Syn[k].STPtr = SynTPtr0[STno]
                        ConMtx0[i][j].Syn[k].wgt = SynPar[4][nst]
                        ConMtx0[i][j].Syn[k].dtax = SynPar[5][nst]
                        ConMtx0[i][j].Syn[k].p_fail = SynPar[6][nst]
                        InList = FALSE
                        kk = 0
                        while (
                                NPtr0[i].STList[kk] is not None and kk < NPtr0[i].NumSynType):
                            if (NPtr0[i].STList[kk].No ==
                                    ConMtx0[i][j].Syn[k].STPtr.No):
                                InList = TRUE
                            kk = kk + 1
                        if (InList == FALSE):
                            NPtr0[i].STList[kk] = ConMtx0[i][j].Syn[k].STPtr
                            NPtr0[i].NumSynType = NPtr0[i].NumSynType + 1
                            NPtr0[i].gfONsyn[kk] = 0.0
                            NPtr0[i].gfOFFsyn[kk] = 0.0
                    else:
                        InList = FALSE
                        kk = 0
                        while (kk < InpNPtr0[j - N].NumPreSyn):
                            if (nst == InpNPtr0[j - N].PreSynList[kk]):
                                InList = TRUE
                                break
                            kk = kk + 1
                        ConMtx0[i][j].Syn[k].PreSynIdx = kk
                        if (InList == FALSE):
                            InpNPtr0[j - N].NumPreSyn = InpNPtr0[j -
                                                                 N].NumPreSyn + 1
                            InpNPtr0[j - N].PreSynList = [0] * \
                                InpNPtr0[j - N].NumPreSyn
                            InpNPtr0[j - N].PreSynList[kk] = nst
                            for num in range(InpNPtr0[j - N].NumPreSyn):
                                InpNPtr0[j - N].SDf.append(SynDepr())
                            InpNPtr0[j - N].SDf[kk].use = SynPar[1][nst]
                            InpNPtr0[j - N].SDf[kk].tc_rec = SynPar[2][nst]
                            InpNPtr0[j - N].SDf[kk].tc_fac = SynPar[3][nst]
                            for k2 in range(SizeHistOutput):
                                InpNPtr0[j - N].SDf[kk].Adepr[k2] = 1.0
                            InpNPtr0[j - N].SDf[kk].uprev[0] = SynPar[1][nst]
                            InpNPtr0[j - N].SDf[kk].Rprev[0] = 1.0
                        STno = int(SynPar[0][nst] - 1)
                        ConMtx0[i][j].Syn[k].STPtr = SynTPtr0[STno]
                        ConMtx0[i][j].Syn[k].wgt = SynPar[4][nst]
                        ConMtx0[i][j].Syn[k].dtax = SynPar[5][nst]
                        ConMtx0[i][j].Syn[k].p_fail = SynPar[6][nst]
                        InList = FALSE
                        kk = 0
                        while (
                                NPtr0[i].STList[kk] is not None and kk < NPtr0[i].NumSynType):
                            if (NPtr0[i].STList[kk].No ==
                                    ConMtx0[i][j].Syn[k].STPtr.No):
                                InList = TRUE
                            kk = kk + 1
                        if (InList == FALSE):
                            NPtr0[i].STList[kk] = ConMtx0[i][j].Syn[k].STPtr
                            NPtr0[i].NumSynType = NPtr0[i].NumSynType + 1
                            NPtr0[i].gfONsyn[kk] = 0.0
                            NPtr0[i].gfOFFsyn[kk] = 0.0
       
        NoiseSyn = SynList()
        NoiseSyn.NumSyn = NumSynType
        NoiseSyn.Syn = []
        for i in range(NoiseSyn.NumSyn):
            NoiseSyn.Syn.append(Synapse())
        for i in range(N):
            for j in range(NoiseSyn.NumSyn):
                STno = int(SynPar[0][numST - NoiseSyn.NumSyn + j] - 1)
                NoiseSyn.Syn[j].STPtr = SynTPtr0[STno]
                NoiseSyn.Syn[j].wgt = SynPar[4][numST - NoiseSyn.NumSyn + j]
                NoiseSyn.Syn[j].dtax = SynPar[5][numST - NoiseSyn.NumSyn + j]
                NoiseSyn.Syn[j].p_fail = SynPar[6][numST - NoiseSyn.NumSyn + j]
                NPtr0[i].gfONnoise[j] = 0.0
                NPtr0[i].gfOFFnoise[j] = 0.0
      
        NoiseStep = 1 / (NoiseDistr.shape[1] - 1)
        
        SynExpOn = [0] * NumSynType
        SynExpOff = [0] * NumSynType
       
        NumEvt = EvtTimes.shape[1]
      
        NumView = ViewList.shape[0] * ViewList.shape[1]
        fpOut = open("IDN_%i.dat" % UniquePrint, "w")
        fpOut2 = open("IDN2_%i.dat" % UniquePrint, "w")
        if (CtrPar[0][3] > NumVar):
            CtrPar[0][3] = NumVar
        NumOutp = 2  
        if (NumOutp > 0):
            NumSynInp = [0] * N
            N_osc = [0] * N
        TnextSyn = [0] * N
        
        
        t0 = Tstart
        time_num = 0
        while (t0 < Tstop):
           
            if (t0 >= t_display):
                print("%f percent" % (t0 * 100 / Tstop))
                t_display = t0 + 100
           
            t1 = t0 + dt0
            EvtNo = -999
            if (t1 > Tstop):
                t1 = Tstop
            for i in range(NumEvt):
                if (EvtTimes[i * 2] > t0) and (EvtTimes[i * 2] <= t1):
                    t1 = EvtTimes[i * 2]
                    NextEvtT = t1
                    EvtNo = i * 2
                else:
                    EvtOffT = EvtTimes[i * 2] + EvtTimes[i * 2 + 1]
                    if (EvtOffT > t0 and EvtOffT <= t1):
                        t1 = EvtOffT
                        NextEvtT = t1
                        EvtNo = i * 2 + 1
            
            t11 = t1
            for i in range(M):
                if (InpNPtr0[i].SPtrain[InpNPtr0[i].SP_ind] > t0) and (
                        InpNPtr0[i].SPtrain[InpNPtr0[i].SP_ind] <= t11):
                    t11 = InpNPtr0[i].SPtrain[InpNPtr0[i].SP_ind]
                    print_flag = 1
                else:
                    print_flag = 0
            t1 = t11
            
            for i in range(M):
                if (InpNPtr0[i].SPtrain[InpNPtr0[i].SP_ind] == t1):
                   
                    if (InpNPtr0[i].SP_ind > 0):
                        ISI_inp = t1 - \
                            InpNPtr0[i].SPtrain[InpNPtr0[i].SP_ind - 1]
                    else:
                        ISI_inp = 10.0e8
                   
                    InpNPtr0[i].SpikeTimes[InpNPtr0[i].SP_ind] = InpNPtr0[i].SPtrain[InpNPtr0[i].SP_ind]
                    InpNPtr0[i].SP_ind = InpNPtr0[i].SP_ind + 1
                   
                    j = NumSpike[i + N] % SizeHistOutput
                   
                    for kk in range(InpNPtr0[i].NumPreSyn):
                        if (InpNPtr0[i].SDf[kk].use > 0.0):
                            InpNPtr0[i].SDf[kk].Adepr[j] = short_time.short_time(
                                SizeHistOutput).syndepr(InpNPtr0[i].SDf[kk], ISI_inp, j)
                    
                    if (WriteST > 0):
                        fpISI = open(
                            "ISIu%d_%i.dat" %
                            (i + N, UniquePrint), "a")
                        fpISI.write("%f\n" % t1)
                        fpISI.close()
                    NumSpike[i + N] = NumSpike[i + N] + 1
            
            for i in range(N):
                
                t0_i = t0
                
                if (t0_i < t1):
                    
                    t1_i = t1
                    
                    if (TnextSyn[i] > t0_i and TnextSyn[i] < t1_i):
                        t1_i = TnextSyn[i]
                    
                    vp = copy.copy(NPtr0[i].v[0])
                    wp = copy.copy(NPtr0[i].v[1])
                    dt = t1_i - t0_i
                    
                    if (NumSpike[i] > 0):
                        if ((t0_i -
                             NPtr0[i].SpikeTimes[(NumSpike[i] -
                                                  1) %
                                                 SizeHistOutput]) < 5):
                            flag_dv = 0
                        else:
                            flag_dv = 1
                    else:
                        flag_dv = 1
                    
                    NPtr0[i] = copy.copy(NPtr0[i])
                    try:
                        NPtr0[i], gsyn_AN, gsyn_G, I_tot = update(
                            NPtr0[i], dt, NoiseSyn, flag_dv)
                    except OverflowError:
                        print(NPtr0[i])
                        print(t0)
                        print(i)
                    if (stop_flag > 0):
                        print("%f %d %f %f\n" % (t0_i, i, vp, wp))
                    for j in range(NPtr0[i].NumSynType):
                        if (NPtr0[i].gfONsyn[j] <
                                0 or NPtr0[i].gfOFFsyn[j] < 0):
                            print(
                                "%d %d %f %f %f %f\n" %
                                (i, j, t0_i, t1_i, NPtr0[i].gfONsyn[j], NPtr0[i].gfOFFsyn[j]))
                    if (t1_i == t1):
                        gsyn1[i] = gsyn_AN
                        gsyn2[i] = gsyn_G
                        Isyn[i] = I_tot
                        if (I_tot < NPtr0[i].I_ref *
                                1.01 and I_tot > NPtr0[i].I_ref * 0.99):
                            flag_osc[i] = flag_osc[i] + 1
                        else:
                            flag_osc[i] = 0
                   
                    if (flag_osc[i] >= 200 and NumOutp > 0):  
                        N_osc[i] = N_osc[i] + 1
                    
                    if ((NPtr0[i].v[0] >= NPtr0[i].Vup)
                            and (vp < NPtr0[i].Vup)):
                        
                        t1_i = t0_i + dt * \
                            (NPtr0[i].Vup - vp) / (NPtr0[i].v[0] - vp)
                        
                        if (NumSpike[i] > 0):
                            ISI = t1_i - \
                                NPtr0[i].SpikeTimes[(NumSpike[i] - 1) % SizeHistOutput]
                        else:
                            ISI = 10.0e8
                       
                        if (ISI > 5):
                            
                            w_Vup = wp + \
                                ((NPtr0[i].v[1] - wp) / dt) * (t1_i - t0_i)
                            NPtr0[i].v[0] = NPtr0[i].Vr
                            NPtr0[i].v[1] = w_Vup + NPtr0[i].b
                            
                            j = NumSpike[i] % SizeHistOutput
                            NPtr0[i].SpikeTimes[j] = t1_i
                           
                            for kk in range(NPtr0[i].NumPreSyn):
                                if (NPtr0[i].SDf[kk].use > 0.0):
                                    NPtr0[i].SDf[kk].Adepr[j] = short_time.short_time(
                                        SizeHistOutput).syndepr(NPtr0[i].SDf[kk], ISI, j)
                            
                            if (WriteST > 0):
                                fpISI = open(
                                    "ISIu%d_%i.dat" %
                                    (i, UniquePrint), "a")
                                fpISI.write("%f\n" % t1_i)
                                fpISI.close()
                            
                            NumSpike[i] = NumSpike[i] + 1
                            
                            dt = t1_i - t0_i
                        else:
                            NPtr0[i].v[0] = vp
                            NPtr0[i].v[1] = wp
                            # reset t1_i
                            t1_i = dt + t0_i
                        
                        if (t1_i == t1):
                            gsyn_AN, I_tot, gsyn_G = short_time.short_time(
                                SizeHistOutput).set_gsyn(NPtr0[i], dt, vp, NoiseSyn)
                            gsyn1[i] = gsyn_AN
                            gsyn2[i] = gsyn_G
                            Isyn[i] = I_tot
                    
                    for j in range(NoiseSyn.NumSyn):
                        SynExpOn[j] = math.exp(-dt /
                                               (NoiseSyn.Syn[j].STPtr).tc_on)
                        SynExpOff[j] = math.exp(-dt /
                                                (NoiseSyn.Syn[j].STPtr).tc_off)
                        rand_num = NoiseDistr[0][rand.randint(
                            0, 1 / NoiseStep)]
                        NPtr0[i].gfONnoise[j] = 0.0
                        NPtr0[i].gfOFFnoise[j] = 0.0
                    
                    for j in range(NPtr0[i].NumSynType):
                        NPtr0[i].gfONsyn[j] *= math.exp(-dt /
                                                        (NPtr0[i].STList[j]).tc_on)
                        NPtr0[i].gfOFFsyn[j] *= math.exp(-dt /
                                                         (NPtr0[i].STList[j]).tc_off)
                    
                    TnextSyn[i] = Tstop + 100.0
                   
                    for j in range(N):
                        
                        for k in range(ConMtx0[i][j].NumSyn):
                            
                            kk = NumSpike[j] - 1
                            while (
                                kk >= 0 and (
                                    NumSpike[j] -
                                    kk) <= SizeHistOutput):
                                if (t0_i >= (
                                        NPtr0[j].SpikeTimes[kk % SizeHistOutput] + ConMtx0[i][j].Syn[k].dtax)):
                                    break
                                else:
                                    
                                    if ((t1_i >= NPtr0[j].SpikeTimes[kk % SizeHistOutput] + ConMtx0[i][j].Syn[
                                            k].dtax) and (
                                            rand.uniform(0, 1) > ConMtx0[i][j].Syn[k].p_fail)):
                                        for k2 in range(NPtr0[i].NumSynType):
                                            if (NPtr0[i].STList[k2].No ==
                                                    ConMtx0[i][j].Syn[k].STPtr.No):
                                                Aall = NPtr0[j].SDf[ConMtx0[i][j].Syn[k].PreSynIdx].Adepr[
                                                    kk % SizeHistOutput] * ConMtx0[i][j].Syn[k].wgt * \
                                                    ConMtx0[i][j].Syn[k].STPtr.Gsyn
                                                NPtr0[i].gfONsyn[k2] += Aall
                                                NPtr0[i].gfOFFsyn[k2] += Aall
                                                if (NumOutp > 0):
                                                    NumSynInp[i] = NumSynInp[i] + 1.0
                                    else:
                                        
                                        if (NPtr0[j].SpikeTimes[kk % SizeHistOutput] +
                                                ConMtx0[i][j].Syn[k].dtax < TnextSyn[i]):
                                            TnextSyn[i] = NPtr0[j].SpikeTimes[kk %
                                                                              SizeHistOutput] + ConMtx0[i][j].Syn[k].dtax
                                kk = kk - 1
                    
                    for j in range(N, N + M):
                        
                        for k in range(ConMtx0[i][j].NumSyn):
                            kk = NumSpike[j] - 1
                            while (kk >= 0):
                               
                                if (t0_i >= (
                                        InpNPtr0[j - N].SpikeTimes[kk] + ConMtx0[i][j].Syn[k].dtax)):
                                    break
                                else:
                                   
                                    if ((t1_i >= InpNPtr0[j - N].SpikeTimes[kk] + ConMtx0[i][j].Syn[k].dtax) and (
                                            rand.uniform(0, 1) > ConMtx0[i][j].Syn[k].p_fail)):
                                        for k2 in range(NPtr0[i].NumSynType):
                                            if (NPtr0[i].STList[k2] ==
                                                    ConMtx0[i][j].Syn[k].STPtr):
                                                Aall = InpNPtr0[j - N].SDf[ConMtx0[i][j].Syn[k].PreSynIdx].Adepr[kk %
                                                                                                                 SizeHistOutput] * ConMtx0[i][j].Syn[k].wgt * (ConMtx0[i][j].Syn[k].STPtr).Gsyn
                                                NPtr0[i].gfONsyn[k2] += Aall
                                                NPtr0[i].gfOFFsyn[k2] += Aall
                                                if (NumOutp > 0):
                                                    NumSynInp[i] = NumSynInp[i] + 1.0
                                    else:
                                        
                                        if (InpNPtr0[j - N].SpikeTimes[kk] + \
                                            ConMtx0[i][j].Syn[k].dtax < TnextSyn[i]):
                                            TnextSyn[i] = InpNPtr0[j - N].SpikeTimes[kk] + \
                                                ConMtx0[i][j].Syn[k].dtax
                                kk = kk - 1
                    
                    t0_i = t1_i
            
            for i in range(NumView):
                fpOut.write("%lf %d" % (t1, ViewList[i][0]))
                for k in range(int(CtrPar[0][3])):
                    fpOut.write(" %lf" % NPtr0[ViewList[i][0] - 1].v[k])
                    fpOut.write("\n")
            
            for i in range(NumView):
                fpOut2.write(" %f %f %f" %
                             (gsyn1[ViewList[i][0] -
                                    1], gsyn2[ViewList[i][0] -
                                              1], Isyn[ViewList[i][0] -
                              1]))
            fpOut2.write("\n")
            
            if (EvtNo >= 0 and t1 >= NextEvtT):
                if ((EvtNo % 2) == 0):
                    for i in range(N):
                        NPtr0[i].Iinj = EvtMtx[int(i + (EvtNo / 2) * N)]
                else:
                    for i in range(N):
                        NPtr0[i].Iinj = 0.0
            
            t0 = t1
        
        if (NumView > 0):
            fpOut.close()
            fpOut2.close()
        
        STMtx = []
        if (CtrPar[0][4] > 0):
            for i in range(N + M):
                if (os.path.exists('ISIu%d_0.dat' % i)):
                    ST = pd.read_table('ISIu%d_0.dat' % i, header=None)
                    content = []
                    for j in range(ST.shape[0]):
                        content.append(ST.iloc[j][0])
                    STMtx.append(content)
                    os.remove('ISIu%d_0.dat' % i)
                else:
                    STMtx.append(-1)
        T = []
        V = []
        if (ViewList is not None):
            X = pd.read_table('IDN_%d.dat' % UniqueNum, header=None)
            for i in range(X.shape[0]):
                T.append(X.iloc[i][0])
                content = []
                for j in range(X.shape[1] - 1):
                    content.append(X.iloc[i][j + 1])
                V.append(content)
            os.remove('IDN_%d.dat' % UniqueNum)
        os.remove('IDN2_0.dat')
        scio.savemat('PFC_%dN_500ms.mat' %
                     (N + M), {'N': N, 'T': T, 'V': V, 'STMtx': STMtx})



class SynType:
    def __init__(self):
        """
        Parameters of short-term synaptic plasticity model
        """
        self.No = 0
        self.gmax = 0
        self.tc_on = 0
        self.tc_off = 0
        self.Erev = 0
        self.Mg_gate = 0
        self.Mg_fac = 0
        self.Mg_slope = 0
        self.Mg_half = 0

        self.Gsyn = 0


class Neuron:
    """
    Parameters of neurons
    """
    gfONsyn = None
    gfOFFsyn = None
    gfONnoise = None
    gfOFFnoise = None
    SpikeTimes = None
    v = None
    dv = None

    def __init__(self):
        MaxNumSTperN = six_layer_pfc().MaxNumSTperN
        SizeHistOutput = six_layer_pfc().SizeHistOutput

        self.Cm = 0
        self.gL = 0
        self.EL = 0
        self.sf = 0
        self.Vup = 0
        self.tcw = 0
        self.a = 0
        self.b = 0
        self.Vr = 0
        self.Vth = 0
        self.I_ref = 0
        self.v_dep = 0
        self.NumSynType = 0

        self.Iinj = 0
        self.v = [0] * 2
        self.dv = [0] * 2
        self.STList = []
        self.gfONsyn = [0] * MaxNumSTperN
        self.gfOFFsyn = [0] * MaxNumSTperN
        self.gfONnoise = [0] * MaxNumSTperN
        self.gfOFFnoise = [0] * MaxNumSTperN
        self.SpikeTimes = [0] * SizeHistOutput
        self.NumPreSyn = 0
        self.PreSynList = []
        self.SDf = []


class InpNeuron:
    """
    Input parameters of neurons
    """
    SPtrain = None
    SpikeTimes = None

    def __init__(self):
        SizeHistInput = six_layer_pfc().SizeHistInput
        self.SPtrain = [0] * SizeHistInput
        self.SpikeTimes = [0] * SizeHistInput
        self.SP_ind = 0
        self.NumSynType = 0
        self.NumPreSyn = 0
        self.PreSynList = []
        self.SDf = []


class Synapse:
    """
    Synaptic parameters
    """
    def __init__(self):
        self.STPtr = SynType()
        self.dtax = 0
        self.wgt = 0
        self.p_fail = 0
        self.PreSynIdx = 0


class SynDepr:
    """
    Parameters of synaptic current model
    """
    Adepr = None
    uprev = None
    Rprev = None

    def __init__(self):
        SizeHistOutput = six_layer_pfc().SizeHistOutput
        self.use = 0
        self.tc_rec = 0
        self.tc_fac = 0
        self.Adepr = [0] * SizeHistOutput
        self.uprev = [0] * SizeHistOutput
        self.Rprev = [0] * SizeHistOutput


class SynList:
    """
    Parameters of synapse list
    """
    def __init__(self):
        self.NumSyn = 0
        self.Syn = []


if __name__ == '__main__':
     """
        After downloading the data file on the network disk, modify the file path to the downloaded placement path
     """
     test = six_layer_pfc()
     path = '/home/duchengcheng/braincog/examples/six_layer_pfc/data100.mat'
     test.mex_function(path)
