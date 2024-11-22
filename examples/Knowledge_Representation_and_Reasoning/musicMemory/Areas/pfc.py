import numpy as np
import math
from braincog.base.brainarea.PFC import PFC
from Modal.synapse import Synapse
from Modal.titlelayer import TitleLayer
from Modal.composerlayer import ComposerLayer
from Modal.genrelayer import GenreLayer
from conf.conf import *
from Modal.layer import *


class PFC(PFC):
    '''
    This area is used to store the sub-Goal of the task
    '''

    def __init__(self, neutype):
        '''
        Constructor
        '''
        super().__init__()
        self.neutype = neutype
        self.goals = TitleLayer(self.neutype)  # store the musical titles
        self.composers = ComposerLayer(self.neutype)  # store composers
        self.keys = KeyLayer(self.neutype)
        self.modes = ModeLayer(self.neutype)
        self.genres = GenreLayer(self.neutype)
        self.chords = ChordLayer(self.neutype)


    def addNewKey(self):
        row, col = configs.key_matrix.shape
        for i in range(row):
            #print(configs.key_matrix[i,:])
            self.keys.addNewGroups(i+1, 1, col, configs.key_matrix[i,:])

    def addNewSubGoal(self, goalname):
        if (self.goals.groups.get(goalname) == None):
            self.goals.addNewGroups(len(self.goals.groups) + 1, 1, 1, goalname)

    def addNewComposer(self, composername):
        if (self.composers.groups.get(composername) == None):
            self.composers.addNewGroups(len(self.composers.groups) + 1, 1, 1, composername)

    def addNewGenre(self, genrename):
        if (self.genres.groups.get(genrename) == None):
            self.genres.addNewGroups(len(self.genres.groups) + 1, 1, 1, genrename)

    def addNewMode(self):
        for i, m in configs.index2mode.items():
            self.modes.addNewGroups(i + 1, 1, 12, m)
            # 与调式网络相连
            scales = configs.keyscales.get(i)
            for k in range(12):  # key neurons project to mode neurons
                for j, index in enumerate(scales[k, :]):
                    pre = self.keys.groups.get(k).neurons[index]
                    post = self.modes.groups.get(i).neurons[j]
                    syn = Synapse(pre, post)
                    syn.excitability = 1
                    syn.type = 3
                    post.synapses.append(syn)
                    post.pre_neurons.append(pre)

                    # syn1 = Synapse(post, pre) # mode neurons project to key neurons
                    # syn1.type = 2
                    # syn1.excitability = 1
                    # syn1.weight = 10 # 这个地方应该设置成KS model
                    # pre.synapses.append(syn1)
                    # pre.pre_neurons.append(post)

    def addNewKey(self):
        row, col = configs.key_matrix.shape
        for i in range(row):
            #print(configs.key_matrix[i,:])
            self.keys.addNewGroups(i+1, 1, col, configs.key_matrix[i,:])

    def addNewChord(self):
        for i in range(0, 7):  # 暂时先存储7个三和弦
            self.chords.addNewGroups(i + 1, 1, 1)
            # 与调式网络相连
            # 先连接T,S,D和弦
            for t, c in configs.chordsMap.items():
                # print(t)
                # print(configs.keyIndexMap.get(t))
                # print(c[i, :])
                # print('---------')
                for k in c[i, :]:
                    pre = self.chords.groups.get(i).neurons[0]
                    post = self.keys.groups.get(configs.keyIndexMap.get(t)).neurons[k]
                    syn = Synapse(pre, post)
                    syn.excitability = 1
                    post.synapses.append(syn)
                    post.pre_neurons.append(pre)
        # 建立和弦内部连接
        r = np.argwhere(configs.chordsMatrix >= 1)
        for i in range(len(r)):
            # print(r[i][0])
            # print(r[i][1])
            pre = self.chords.groups.get(r[i][0]).neurons[0]
            post = self.chords.groups.get(r[i][1]).neurons[0]
            syn = Synapse(pre, post)
            post.synapses.append(syn)
            post.pre_neurons.append(pre)

    def setTestStates(self):

        self.goals.setTestStates()
        self.composers.setTestStates()
        self.genres.setTestStates()
        self.keys.setTestStates()
        self.modes.setTestStates()
        self.chords.setTestStates()

    def doRecalling(self, goalname, asm):
        goal = self.goals.groups.get(goalname)
        #         print(goal.name)
        #         print(goal.id)
        result = {}
        sequences = asm.sequenceLayers.get(1).groups
        dt = 0.1
        time = np.arange(0, len(sequences) * 5, dt)

        for t in time:
            order = math.floor(t / 5) + 1

            for neu in goal.neurons:
                neu.I = 30
                neu.update_normal(dt, t)
            sg = sequences.get(order)
            for neu in sg.neurons:
                neu.updateCurrentOfLowerAndUpperLayer(t)
                neu.update(dt, t, 'test')
                # if(neu.spike == True):
                # print(neu.index)
                if (neu.spike == True and result.get(order) == None):
                    result[int(order)] = neu.selectivity
        return result

    def doRecalling2(self, goalname, asm):
        goal = self.goals.groups.get(goalname)
        #         print(goal.name)
        #         print(goal.id)

        result = {}
        for tindex, strack in asm.sequenceLayers.items():
            nsequences = strack.get("N").groups
            tsequences = strack.get("T").groups
            dic = {}
            ndic = {}
            tdic = {}
            dt = 0.1
            time = np.arange(0, len(nsequences) * 5, dt)
            for t in time:
                order = math.floor(t / 5) + 1
                for neu in goal.neurons:
                    neu.I = 30
                    neu.update_normal(dt, t)
                nsg = nsequences.get(order)
                for neu in nsg.neurons:
                    # print(neu.selectivity)
                    neu.updateCurrentOfLowerAndUpperLayer(t)
                    neu.update(dt, t, 'test')
                    # if(neu.I > 0):
                    #     print(neu.I)
                    if (neu.spike == True and ndic.get(order) == None):
                        ndic[int(order)] = neu.selectivity

                tsg = tsequences.get(order)
                for neu in tsg.neurons:
                    neu.updateCurrentOfLowerAndUpperLayer(t)
                    neu.update(dt, t, 'test')
                    if (neu.spike == True and tdic.get(order) == None):
                        tdic[int(order)] = neu.selectivity

            dic["N"] = ndic
            dic["T"] = tdic
            result[tindex] = dic
        return result

    def doRecalling3(self,goalname,asm):
        goal = self.goals.groups.get(goalname)
        #         print(goal.name)
        #         print(goal.id)

        result = {}
        for tindex, strack in asm.sequenceLayers.items():
            nsequences = strack.get("N").groups
            tsequences = strack.get("T").groups
            part = []
            ndic = {}
            tdic = {}
            dt = 0.1
            # for order in range(len(nsequences)):
            #     nns = nsequences.get(order+1).neurons
            #     for n in nns:
            #         if n.preActive:
            #             print('-------order:'+str(order)+', selectivity: '+str(n.selectivity)+'--------------------')
            #             for syn in n.synapses:
            #                 if syn.weight > 0:
            #                     print(syn.weight)
            #time = np.arange(0, len(nsequences) * 5, dt)
            print(len(nsequences))
            for i in range(0,len(nsequences)):
                order = i+1
                tmp = {}
                time = np.arange(i*5,(i+1)*5,dt)
                for t in time:
                    for neu in goal.neurons:
                        neu.I = 30
                        neu.update_normal(dt, t)
                    nsg = nsequences.get(order)
                    for neu in nsg.neurons:
                        neu.updateCurrentOfLowerAndUpperLayer(t)
                        neu.update_normal(dt, t)
                        # if (neu.I > 0):
                        #     print('order: ' + str(order))
                        #     print(neu.I)
                        #     print(neu.selectivity)
                        #     print(neu.I_lower)
                        if (neu.spike == True and ndic.get(order) == None):# 这里用的是first spike的理念，我觉得最好改成max firingrate
                            ndic[int(order)] = neu.selectivity
                            tmp["N"] = neu.selectivity

                    tsg = tsequences.get(order)
                    for neu in tsg.neurons:
                        neu.updateCurrentOfLowerAndUpperLayer(t)
                        neu.update_normal(dt, t)
                        if (neu.spike == True and tdic.get(order) == None):#这个地方bug太邪乎了，等一会儿改
                            tdic[int(order)] = neu.selectivity
                            tmp["T"] = neu.selectivity
                part.append(tmp)
            result[tindex] = part
        print(len(result))
        return result

    def doRemebering(self, goalname, dt, t):
        # storing the title information
        goal_group = self.goals.groups.get(goalname)
        for neu in goal_group.neurons:
            neu.I = 50
            neu.update_normal(dt, t)

    def doRememberingComposer(self, composername, dt, t):
        composer_group = self.composers.groups.get(composername)
        for neu in composer_group.neurons:
            neu.I = 50
            neu.update_normal(dt, t)

    def doRememberingGenre(self, genrename, dt, t):
        genre_group = self.genres.groups.get(genrename)
        for neu in genre_group.neurons:
            neu.I = 50
            neu.update_normal(dt, t)

    def doRememberingKey(self, key, dt, t):
        key_group = self.keys.groups.get(key)
        for neu in key_group.neurons:
            neu.I = 50 if neu.importance > 0 else -100
            neu.update_learn(dt, t)

    def doRememberingMode(self,mode, dt,t):
        mode_group = self.modes.groups.get(mode)
        for neu in mode_group.neurons:
            neu.I = 50
            neu.update_learn(dt,t)

    def innerLearning(self, goalname, composer, genre):
        g = self.goals.groups.get(goalname)
        c = self.composers.groups.get(composer)
        gre = self.genres.groups.get(genre)
        if (g != None and c != None):
            for n1 in c.neurons:
                if (len(n1.spiketime) > 0):
                    for n2 in g.neurons:
                        if (len(n2.spiketime) > 0):
                            temp = 0
                            for sp1 in n1.spiketime:
                                for sp2 in n2.spiketime:
                                    if (abs(sp1 - sp2) <= n1.tau_ref):
                                        temp += 1
                            if (temp >= 3):
                                syn = Synapse(n1, n2)
                                syn.type = 2
                                syn.weight = 5
                                n2.synapses.append(syn)
                                n2.pre_neurons.append(n1)

        if (gre != None):
            for n1 in gre.neurons:
                if (len(n1.spiketime) > 0):
                    if (c != None):
                        for n2 in c.neurons:
                            if (len(n2.spiketime) > 0):
                                temp = 0
                                for sp1 in n1.spiketime:
                                    for sp2 in n2.spiketime:
                                        if (abs(sp1 - sp2) <= n1.tau_ref):
                                            temp += 1
                                if (temp >= 4):
                                    syn = Synapse(n1, n2)
                                    syn.type = 2
                                    syn.weight = 5
                                    n2.synapses.append(syn)
                                    n2.pre_neurons.append(n1)
                    if (g != None):
                        for n2 in g.neurons:
                            if (len(n2.spiketime) > 0):
                                temp = 0
                                for sp1 in n1.spiketime:
                                    for sp2 in n2.spiketime:
                                        if (abs(sp1 - sp2) <= n1.tau_ref):
                                            temp += 1
                                if (temp >= 4):
                                    syn = Synapse(n1, n2)
                                    syn.type = 2
                                    syn.weight = 5
                                    n2.synapses.append(syn)
                                    n2.pre_neurons.append(n1)

    def inhibitGenres(self,dt,t):
        gen_group = self.goals.groups
        for g in gen_group.values():
            for neu in g.neurons:
                neu.I = -100
                neu.update_normal(dt, t)

    def inhibiteGoals(self, dt, t):
        goal_group = self.goals.groups
        for g in goal_group.values():
            for neu in g.neurons:
                neu.I = -100
                neu.update(dt, t)

    def inhibitComposers(self, dt, t):
        com_group = self.composers.groups
        for g in com_group.values():
            for neu in g.neurons:
                neu.I = -100
                neu.update(dt, t)







