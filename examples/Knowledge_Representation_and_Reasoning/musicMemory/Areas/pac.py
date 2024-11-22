'''
Primary Auditory Area
'''

from braincog.base.brainarea.BrainArea import BrainArea

from Modal.sequencememory import SequenceMemory
from Modal.notesequencelayer import NoteSequenceLayer
from Modal.temposequencelayer import TempoSequenceLayer
from Modal.synapse import Synapse
import numpy as np
import math
from conf.conf import *


class PAC(BrainArea,SequenceMemory):
    '''
     the planum polare, anterior to PAC, as well as in the left planum temporale,posterior to PAC.
    '''

    def __init__(self, neutype):
        '''
        Constructor
        '''
        SequenceMemory.__init__(self, neutype)

    def forward(self, x):
        pass

    def createActionSequenceMem(self, layernum, neutype):

        sl = NoteSequenceLayer(neutype)
        tl = TempoSequenceLayer(neutype)
        instrumentTrack = {}
        instrumentTrack["N"] = sl
        instrumentTrack["T"] = tl
        self.sequenceLayers[layernum] = instrumentTrack
        print(len(self.sequenceLayers))

    def doRemembering_note_only(self, note, order, dt, t):
        # remember note
        sl = self.sequenceLayers.get(1)
        sgroup = sl.groups.get(order)
        dt = 0.1
        for n in sgroup.neurons:
            n.I_ext = note.frequence
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')

    def doRemembering(self, trackIndex, noteIndex, order, dt, t, tinterval=0):
        # remember note
        iTrack = self.sequenceLayers.get(trackIndex)
        sl = iTrack.get("N")
        sgroup = sl.groups.get(order)
        dt = 0.1
        for n in sgroup.neurons:
            n.I_ext = noteIndex
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')

        # remember tempo
        tl = iTrack.get("T")
        tgroup = tl.groups.get(order)
        dt = 0.1
        for n in tgroup.neurons:
            n.I_ext = tinterval
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')



    def doConnectToTitle(self, title, track, order):
        for sl in track.values():
            self.doConnecting(title, sl, order)

    def doConnectToComposer(self, composer, track, order):
        for sl in track.values():
            self.doConnecting(composer, sl, order)

    def doConnectToGenre(self, genre, track, order):
        for sl in track.values():
            self.doConnecting(genre, sl, order)

    def generateEx_Nihilo(self, firstNote, durations, order, dt, t):
        ns = self.sequenceLayers.get(1).get("N")
        ts = self.sequenceLayers.get(1).get("T")
        nneurons = ns.groups.get(order + 1).neurons
        tneurons = ts.groups.get(order + 1).neurons
        # firstNotes specify the beginning notes to trigger the following notes
        if (order < len(firstNote)):  # beginning notes
            i = firstNote[order]
            nneu = nneurons[i + 1]
            nneu.I = 20
            nneu.update_normal(dt, t)

            d = int(durations[order] / 0.125) - 1
            tneu = tneurons[d]
            tneu.I = 20

            tneu.update_normal(dt, t)
        else:  # generate next note
            for nn in nneurons:
                nn.updateCurrentOfLowerAndUpperLayer(t)
                nn.update(dt, t, 'test')
            for tn in tneurons:
                tn.updateCurrentOfLowerAndUpperLayer(t)
                tn.update(dt, t, 'test')

    def generateSimgleTrackNotes(self, trackIndex, firstNote, durations, order, dt, t):
        ns = self.sequenceLayers.get(trackIndex).get("N")
        ts = self.sequenceLayers.get(trackIndex).get("T")
        nneurons = ns.groups.get(order + 1).neurons
        tneurons = ts.groups.get(order + 1).neurons
        # firstNotes specify the beginning notes to trigger the following notes
        if (order < len(firstNote)):  # beginning notes
            i = firstNote[order]
            nneu = nneurons[i + 1]
            nneu.I = 20
            nneu.update_normal(dt, t)

            d = int(durations[order] / 0.125) - 1
            tneu = tneurons[d]
            tneu.I = 20
            tneu.update_normal(dt, t)
        else:  # generate next note
            for nn in nneurons:
                nn.updateCurrentOfLowerAndUpperLayer(t)
                nn.update(dt, t, 'test')
            #                 if(neu.spike == True):
            #                     print(neu.selectivity)
            for tn in tneurons:
                tn.updateCurrentOfLowerAndUpperLayer(t)
                tn.update(dt, t, 'test')


class Music_Sequence_Mem(SequenceMemory):
    '''
     the planum polare, anterior to PAC, as well as in the left planum temporale,posterior to PAC.
    '''

    def __init__(self, neutype):
        '''
        Constructor
        '''
        SequenceMemory.__init__(self, neutype)

    def createActionSequenceMem(self, layernum, neutype):

        sl = NoteSequenceLayer(neutype)
        tl = TempoSequenceLayer(neutype)
        instrumentTrack = {}
        instrumentTrack["N"] = sl
        instrumentTrack["T"] = tl
        self.sequenceLayers[layernum] = instrumentTrack
        print(len(self.sequenceLayers))

    def doRemembering_note_only(self, note, order, dt, t):
        # remember note
        sl = self.sequenceLayers.get(1)
        sgroup = sl.groups.get(order)
        dt = 0.1
        for n in sgroup.neurons:
            n.I_ext = note.frequence
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')

    def doRemembering(self, trackIndex, noteIndex, order, dt, t, tinterval=0):
        # remember note
        iTrack = self.sequenceLayers.get(trackIndex)
        sl = iTrack.get("N")
        sgroup = sl.groups.get(order)
        dt = 0.1
        for n in sgroup.neurons:
            n.I_ext = noteIndex
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')

        # remember tempo
        tl = iTrack.get("T")
        tgroup = tl.groups.get(order)
        dt = 0.1
        for n in tgroup.neurons:
            n.I_ext = tinterval
            n.computeFilterCurrent()
            n.update(dt, t, 'Learn')

    def recallByEpisode(self, episodeNotes, goals):
        dt = 0.1
        sl = self.sequenceLayers.get(1)
        firstresult = {}
        firstNote = episodeNotes[0]
        tt = np.arange(0, 5, dt)

        # find first note and activate goal neurons
        for t in tt:
            for id, group in sl.groups.items():
                neuid = firstNote - 15;
                if (group.neurons[neuid - 1].preActive == False): continue
                group.neurons[neuid - 1].I_ext = 20
                group.neurons[neuid - 1].updateCurrentOfLowerAndUpperLayer(t)
                group.neurons[neuid - 1].update(dt, t, 'test')
                if (group.neurons[neuid - 1].spike == True):
                    firstresult[id] = 1
            for name, g in goals.groups.items():
                for neu in g.neurons:
                    neu.updateCurrentOfLowerAndUpperLayer(t)
                    neu.update(dt, t)

        # find rest episode notes
        restResult = {}
        # for i in range(1,len(episodeNotes)):
        goalchecked = {}
        for groupID, value in firstresult.items():
            tmp = {}
            for i in range(1, len(episodeNotes)):
                fre = episodeNotes[i]
                tt = np.arange(i * 5, (i + 1) * 5, dt)
                g = sl.groups.get(groupID + i)
                for t in tt:
                    # update memory
                    neuid = fre - 15;
                    if (g.neurons[neuid - 1].preActive == False): continue
                    g.neurons[neuid - 1].I_ext = 20
                    g.neurons[neuid - 1].updateCurrentOfLowerAndUpperLayer(t)
                    if (g.neurons[neuid - 1].I_lower == 0): break
                    g.neurons[neuid - 1].update(dt, t, 'test')
                    if (g.neurons[neuid - 1].spike == True):
                        tmp[i] = g.id

                    # update goals' neurons
                    for name, gg in goals.groups.items():
                        for neu in gg.neurons:
                            neu.updateCurrentOfLowerAndUpperLayer(t)
                            neu.update(dt, t)
                            # if(neu.spike == True):print(name)
            # find
            maxFiringRate = 0
            maxGoalName = ''
            maxGoal = {}  # an episode may be mapped to more than one songs
            for name, gg in goals.groups.items():
                if (goalchecked.get(name) == None):
                    averageFiringRate = 0
                    for neu in gg.neurons:
                        averageFiringRate = averageFiringRate + len(neu.spiketime)
                    averageFiringRate = float(averageFiringRate) / float(len(gg.neurons))
                    gg.averageFiringRate = averageFiringRate
                    if (averageFiringRate > maxFiringRate):
                        maxFiringRate = averageFiringRate
                        maxGoalName = name
            maxGoal[maxGoalName] = 1
            goalchecked[maxGoalName] = 1
            for name, gg in goals.groups.items():
                if (gg.averageFiringRate == maxFiringRate and goalchecked.get(name) == None):
                    maxGoal[name] = 1
                    goalchecked[name] = 1
            tmp['goal'] = maxGoal
            restResult[groupID] = tmp
        print(restResult)
        episodeResult = []
        for key, value in firstresult.items():
            tmp = {}
            tmp[0] = key
            dic = restResult.get(key)
            for i in range(1, len(episodeNotes)):
                if (dic.get(i) != None):
                    tmp[i] = dic.get(i)
            if (len(tmp) == len(episodeNotes)):
                tmp['goal'] = dic.get('goal')
                episodeResult.append(tmp)
        print(episodeResult)

        for res in episodeResult:
            msmgroupID = res.get(len(episodeNotes) - 1) + 1
            for gname, value in res.get('goal').items():
                gg = goals.groups.get(gname)

    def recallByEpisode2(self, episodeNotes, goals):
        dt = 0.1
        sl = self.sequenceLayers.get(1).get("N")
        tl = self.sequenceLayers.get(1).get("T")
        print(len(sl.groups))
        firstresult = {}
        firstNote = episodeNotes[0]
        tt = np.arange(0, 5, dt)

        # find first note and activate goal neurons
        for t in tt:
            for id, group in sl.groups.items():
                neuid = firstNote;
                if (group.neurons[neuid + 1].preActive == False): continue
                # group.neurons[neuid+1].I_ext = 20
                # group.neurons[neuid+1].updateCurrentOfLowerAndUpperLayer(t)
                group.neurons[neuid + 1].I = 20
                group.neurons[neuid + 1].update(dt, t, 'test')
                if (group.neurons[neuid + 1].spike == True):
                    firstresult[id] = 1

        # find rest episode notes
        restResult = {}
        # for i in range(1,len(episodeNotes)):
        goalchecked = {}
        for groupID in firstresult.keys():
            #
            sl.setTestStates()

            tmp = {}
            tt = np.arange(0, 5, dt)
            g = sl.groups.get(groupID)
            neuid = firstNote
            # g.neurons[neuid+1].I_ext = 20
            g.neurons[neuid + 1].I = 20
            for t in tt:
                # g.neurons[neuid+1].updateCurrentOfLowerAndUpperLayer(t)
                g.neurons[neuid + 1].update(dt, t, 'test')

            for i in range(1, len(episodeNotes)):
                fre = episodeNotes[i]
                tt = np.arange(i * 5, (i + 1) * 5, dt)
                if (groupID + i > len((sl.groups))): continue
                g = sl.groups.get(groupID + i)
                for t in tt:
                    # update memory
                    neuid = fre;
                    if (g.neurons[neuid + 1].preActive == False): continue
                    g.neurons[neuid + 1].I_ext = 20
                    g.neurons[neuid + 1].updateCurrentOfLowerAndUpperLayer(t)
                    if (g.neurons[neuid + 1].I_lower == 0): break
                    g.neurons[neuid + 1].update(dt, t, 'test')
                    if (g.neurons[neuid + 1].spike == True):
                        tmp[i] = g.id

            restResult[groupID] = tmp
            # print(restResult)
        episodeResult = []
        for key in firstresult.keys():
            tmp = {}
            tmp[0] = key
            dic = restResult.get(key)
            for i in range(1, len(episodeNotes)):
                if (dic.get(i) != None):
                    tmp[i] = dic.get(i)
            if (len(tmp) == len(episodeNotes)):
                episodeResult.append(tmp)
        # print(episodeResult)

        # begin remembering
        finalResult = []
        for i, res in enumerate(episodeResult):
            goals.setTestStates()
            self.setTestStates()
            for i, fre in enumerate(episodeNotes):
                tt = np.arange(i * 5, (i + 1) * 5, dt)
                neuid = fre
                g = sl.groups.get(res.get(i))
                for t in tt:
                    # g.neurons[neuid+1].I_ext = 20
                    # g.neurons[neuid+1].updateCurrentOfLowerAndUpperLayer(t)
                    g.neurons[neuid + 1].I = 20
                    g.neurons[neuid + 1].update(dt, t, 'test')

                    for name, gg in goals.groups.items():
                        for neu in gg.neurons:
                            neu.updateCurrentOfLowerAndUpperLayer(t)
                            neu.update(dt, t)
            # find goal
            maxFiringRate = 0
            maxGoalName = ''
            maxGoal = {}  # an episode may be mapped to more than one songs
            for name, gg in goals.groups.items():
                averageFiringRate = 0
                for neu in gg.neurons:
                    averageFiringRate = averageFiringRate + len(neu.spiketime)
                averageFiringRate = float(averageFiringRate) / float(len(gg.neurons))
                gg.averageFiringRate = averageFiringRate
                if (averageFiringRate > maxFiringRate):
                    maxFiringRate = averageFiringRate
                    maxGoalName = name
            maxGoal[maxGoalName] = 1
            for name, gg in goals.groups.items():
                if (gg.averageFiringRate == maxFiringRate and goalchecked.get(name) == None):
                    maxGoal[name] = 1
            # print(maxGoal)

            # recall the rest song
            for goalname, value in maxGoal.items():
                # reset State
                goals.setTestStates()
                nextGroupId = res.get(len(episodeNotes) - 1) + 1
                for i in range(nextGroupId, len(sl.groups) + 1):
                    sl.groups.get(i).setTestStates()
                    tl.groups.get(i).setTestStates()
                restSpikeResult = {}
                count = 0
                gg = goals.groups.get(goalname)
                for i in range(nextGroupId, len(sl.groups) + 1):
                    order = len(episodeNotes) + count
                    if (order == 3):
                        print("debug")
                    tt = np.arange(order * 5, (order + 1) * 5, dt)
                    msmgroup = sl.groups.get(i)
                    msmtgroup = tl.groups.get(i)
                    tdic = {}
                    for t in tt:
                        for n in gg.neurons:
                            # n.updateCurrentOfLowerAndUpperLayer(t)
                            n.I = 30
                            n.update_normal(dt, t)

                        for neu in msmgroup.neurons:
                            neu.updateCurrentOfLowerAndUpperLayer(t)
                            neu.update(dt, t, 'test')
                            if (neu.spike == True and restSpikeResult.get(order) == None):
                                # restSpikeResult[int(order)] = neu.selectivity
                                tdic["N"] = neu.selectivity

                        for neu in msmtgroup.neurons:
                            neu.updateCurrentOfLowerAndUpperLayer(t)
                            neu.update(dt, t, 'test')
                            if (neu.spike == True and restSpikeResult.get(order) == None):
                                # restSpikeResult[int(order)] = neu.selectivity
                                tdic["T"] = neu.selectivity
                    if (tdic):
                        restSpikeResult[int(order)] = tdic
                    count += 1
                # print(restSpikeResult)
                dic = {}
                dic['goal'] = goalname
                dic['rest'] = restSpikeResult
                finalResult.append(dic)
        return (finalResult)

    def doConnectToTitle(self, title, track, order):
        for sl in track.values():
            self.doConnecting(title, sl, order)

    def doConnectToComposer(self, composer, track, order):
        for sl in track.values():
            self.doConnecting(composer, sl, order)

    def doConnectToGenre(self, genre, track, order):
        for sl in track.values():
            self.doConnecting(genre, sl, order)

    def doConnectToEmotion(self, emo, track, order):
        for sl in track.values():
            self.doConnecting(emo, sl, order)

    def doConnectToKey(self, key, track, order, noteIndex):
        group = track.get("N").groups.get(order)
        if group == None: return
        tb = (order - 1) * group.timeWindow
        te = (order) * group.timeWindow
        sp1_goal = {}
        sp2 = []
        kneu = key.neurons[noteIndex % 12]
        sp = []
        for st in kneu.spiketime:
            if (st < te and st >= tb):
                sp.append(st)
        sp1_goal[kneu.index] = sp

        n = group.neurons[noteIndex + 1]
        if (len(n.spiketime) > 0):
            for index, sp in sp1_goal.items():
                temp = 0
                for sp1 in n.spiketime:  # spike times of group
                    for sp2 in sp:
                        if (abs(sp1 - sp2) <= n.timeWindow):
                            temp += 1
                if (temp >= 2):  # super threshold, create a new synapse between goal and neurons of sequence group
                    # syn = Synapse(goal.neurons[index - 1], n)
                    # syn.type = 2
                    # syn.weight = 30
                    # n.pre_neurons.append(goal.neurons[index - 1])
                    # n.synapses.append(syn)

                    # add reverse synapse to neurons of the goal
                    syn2 = Synapse(n, kneu)
                    syn2.type = 3
                    syn2.weight = 0
                    kneu.synapses.append(syn2)

    def doConnectToMode(self, mode, keyName, track, order, noteIndex):  # 只连接对应调式的音级(这是两个神经元之间的连接)
        noteScales = np.where((configs.keyscales.get(keyName % 2)[keyName // 2]) == noteIndex % 12)[0][0]
        group = track.get("N").groups.get(order)
        if (group == None): return
        tb = (order - 1) * group.timeWindow
        te = (order) * group.timeWindow
        sp1_goal = {}
        sp2 = []

        modeneu = mode.neurons[noteScales]
        sp = []
        for st in modeneu.spiketime:
            if (st < te and st >= tb):
                sp.append(st)
        sp1_goal[modeneu.index] = sp

        n = group.neurons[noteIndex + 1]
        if (len(n.spiketime) > 0):
            for index, sp in sp1_goal.items():
                temp = 0
                for sp1 in n.spiketime:  # spike times of group
                    for sp2 in sp:
                        if (abs(sp1 - sp2) <= n.timeWindow):
                            temp += 1
                if (temp >= 2):  # super threshold, create a new synapse between goal and neurons of sequence group
                    # syn = Synapse(goal.neurons[index - 1], n)
                    # syn.type = 2
                    # syn.weight = 30
                    # n.pre_neurons.append(goal.neurons[index - 1])
                    # n.synapses.append(syn)

                    # add reverse synapse to neurons of the goal
                    syn2 = Synapse(n, modeneu)
                    syn2.type = 3
                    syn2.weight = 0
                    modeneu.synapses.append(syn2)

    def generateEx_Nihilo(self, firstNote, durations, order, dt, t):
        ns = self.sequenceLayers.get(1).get("N")
        ts = self.sequenceLayers.get(1).get("T")
        nneurons = ns.groups.get(order + 1).neurons
        tneurons = ts.groups.get(order + 1).neurons
        # firstNotes specify the beginning notes to trigger the following notes
        if (order < len(firstNote)):  # beginning notes
            i = firstNote[order]
            nneu = nneurons[i + 1]
            nneu.I = 20
            nneu.update_normal(dt, t)

            d = int(durations[order] / 0.125) - 1
            tneu = tneurons[d]
            tneu.I = 20

            tneu.update_normal(dt, t)
        else:  # generate next note
            for nn in nneurons:
                nn.updateCurrentOfLowerAndUpperLayer(t)
                nn.update(dt, t, 'test')
            for tn in tneurons:
                tn.updateCurrentOfLowerAndUpperLayer(t)
                tn.update(dt, t, 'test')

    def generateMelodyWithTone(self, firstNote, duration, tone, order, dt, t):
        for i, part in self.sequenceLayers.items():
            if i > 4: break
            ns = part.get("N")
            ts = None
            if duration is not None:
                ts = part.get("T")  # 不输入值的话就都生成四分音符

            # pitches updating
            if firstNote is not None:
                for nneu in ns.groups.get(order).neurons:
                    nneu.I_ext = firstNote[i - 1]
                    nneu.computeFilterCurrent()
                    nneu.update_normal(dt, t)
            else:
                for nneu in ns.groups.get(order).neurons:
                    nneu.updateCurrentOfLowerAndUpperLayer(t)
                    nneu.update_normal(dt, t)

            # durations updating
            if ts is not None:
                if duration is not None:
                    for tneu in ts.groups.get(order).neurons:
                        tneu.I_ext = duration[i - 1]
                        tneu.computeFilterCurrent()
                        tneu.update_normal(dt, t)
                else:
                    for tneu in ts.groups.get(order).neurons:
                        tneu.updateCurrentOfLowerAndUpperLayer(t)
                        tneu.update_normal(dt, t)

    def generateSimgleTrackNotes(self, trackIndex, firstNote, durations, order, dt, t):
        ns = self.sequenceLayers.get(trackIndex).get("N")
        ts = self.sequenceLayers.get(trackIndex).get("T")
        nneurons = ns.groups.get(order + 1).neurons
        tneurons = ts.groups.get(order + 1).neurons
        # firstNotes specify the beginning notes to trigger the following notes
        if (order < len(firstNote)):  # beginning notes
            i = firstNote[order]
            nneu = nneurons[i + 1]
            nneu.I = 20
            nneu.update_normal(dt, t)

            d = int(durations[order] / 0.125) - 1
            tneu = tneurons[d]
            tneu.I = 20
            tneu.update_normal(dt, t)
        else:  # generate next note
            for nn in nneurons:
                nn.updateCurrentOfLowerAndUpperLayer(t)
                nn.update(dt, t, 'test')
            #                 if(neu.spike == True):
            #                     print(neu.selectivity)
            for tn in tneurons:
                tn.updateCurrentOfLowerAndUpperLayer(t)
                tn.update(dt, t, 'test')