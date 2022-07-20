'''
Primary Auditory Area
'''

from braincog.base.brainarea.BrainArea import BrainArea

from Modal.sequencememory import SequenceMemory
from Modal.notesequencelayer import NoteSequenceLayer
from Modal.temposequencelayer import TempoSequenceLayer
import numpy as np
import math


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