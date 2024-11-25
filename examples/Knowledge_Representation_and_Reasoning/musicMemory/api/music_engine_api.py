

from conf.conf import *
from Areas.cortex import Cortex
import pretty_midi
import math
import json
import music21 as m21


class EngineAPI():
    '''

    '''

    def __init__(self):
        '''
        Constructor
        '''

        self.cortex = Cortex(configs.neuron_type, configs.dt)

    def cortexInit(self):
        self.cortex.musicSequenceMemroyInit()
        self.cortex.pfc.addNewKey()
        self.cortex.pfc.addNewMode()
        self.cortex.pfc.addNewChord()

    def rememberMusic(self, muiscName, composerName="None"):
        '''
        :param muiscName: the name of the melody
        :param composerName: the composer
        :return:
        '''
        muiscName = muiscName.title()
        composerName = composerName.title()
        self.cortex.pfc.setTestStates()
        self.cortex.msm.setTestStates()
        self.cortex.addSubGoalToPFC(muiscName)
        self.cortex.addComposerToPFC(composerName)
        genreName = str(configs.GenreMap.get(composerName))
        self.cortex.addGenreToPFC(genreName)
        self.cortex.pfc.innerLearning(muiscName, composerName, genreName)

        goaldic = {}
        composerdic = {}
        genredic = {}
        if (configs.RunTimeState == 1):
            g = self.cortex.pfc.titles.groups.get(muiscName)
            c = self.cortex.pfc.composers.groups.get(composerName)
            gre = self.cortex.pfc.genres.groups.get(genreName)
            goaldic = g.writeSelfInfoToJson("IPS")
            composerdic = c.writeSelfInfoToJson("Composer")
            genredic = gre.writeSelfInfoToJson("Genre")

        return goaldic, composerdic

    def learnFourPartMusic(self,xmldata, musicName, composerName="None"):
        musicName = musicName.title()
        composerName = composerName.title()
        genreName = "None"
        toneName = configs.keyIndexMap.get(str(xmldata.analyze('key')))

        print(musicName + " learning...")

        emo = "None"
        for i, part in enumerate(xmldata.parts):
            if (self.cortex.msm.sequenceLayers.get(i + 1) == None):
                self.cortex.msm.createActionSequenceMem(i + 1, self.cortex.neutype)
            self.rememberPartNotes(musicName, composerName, genreName, emo, toneName, i + 1, part)


    def rememberPartNotes(self,musicName, composerName, genreName, emo, keyName, partIndex, part):
        print("Learning the part "+str(partIndex))
        for i,note in enumerate(part.flat.notes[:20]):
            p = 0
            dur = 0
            if note.isChord:
                dur = note.duration.quarterLength
                for chord_note in note:
                    p = m21.pitch.Pitch(chord_note.pitch).midi

            else:
                dur = note.duration.quarterLength
                p = m21.pitch.Pitch(note.pitch).midi
            if dur == 0.0:
                dur = 0.125
            if keyName == 'None':
                self.cortex.rememberANoteandTempo(musicName, composerName, genreName, emo, partIndex, p, i+1, dur)
            else:
                self.cortex.rememberANoteWithKnowledge(musicName, composerName, genreName, emo, keyName, partIndex, p, dur, i+1, part)

    def rememberMIDIMusic(self, musicName, composerName, noteLength, fileName):
        '''
        :param musicName: the name of the piece of music
        :param composerName: the composer who writes this melody
        :param fileName: the name of this midi file
        :return: none
        '''
        musicName = musicName.title()
        composerName = composerName.title()
        print(musicName + " processing...")
        pm = pretty_midi.PrettyMIDI(fileName)
        genreName = str(configs.GenreMap.get(composerName))
        for i, ins in enumerate(pm.instruments):
            if (i >= 1): break;
            if (self.cortex.msm.sequenceLayers.get(i + 1) == None):
                # create a new layer to store the track
                self.cortex.msm.createActionSequenceMem(i + 1, self.cortex.neutype)
            self.rememberTrackNotes(musicName, composerName, genreName, i + 1, ins, pm, noteLength)
        print(musicName + " finished!")

    def rememberTrackNotes(self, musicName, composerName, genreName, trackIndex, track, pm, noteLength):
        r_notes = []
        r_intervals = []
        total_dic = {}

        print(track)
        if(noteLength == "ALL"):
            noteLength = len(track.notes)
        order = 1
        i = 0
        #while (i < len(track.notes)):
        while (i < noteLength):
            #if (i >= rl): break;
            note = track.notes[i]
            start = pm.time_to_tick(note.start)
            end = pm.time_to_tick(note.end)
            pitches = []
            durations = []
            restFlag = False
            # this part recognizes a rest
            if (i == 0):  # determine whether the first note is a rest
                if (start >= 30):
                    pitches.append(-1)  # -1 represents a rest
                    durations.append(start / pm.resolution)
                    restFlag = True
            else:
                lastend = pm.time_to_tick(track.notes[i - 1].end)
                if (start - lastend >= 50):
                    pitches.append(-1)
                    durations.append((start - lastend) / pm.resolution)
                    restFlag = True
            if (restFlag == True):
                dic, g = self.rememberANote(musicName, composerName, genreName, trackIndex, pitches[0], order,
                                            durations[0], True)
                if (configs.RunTimeState == 1):
                    jstr = json.dumps(g)
                    self.conn.send('/Queue/SampleQueue', jstr)
                #print(str(order) + ":(-1," + str(durations[0]) + ")")
                order = order + 1
                pitches = []
                durations = []

                # this part recognizes a chord
            pitches.append(note.pitch)
            durations.append((end - start) / pm.resolution)
            j = i + 1
            while (j < len(track.notes)):
                nextstart = pm.time_to_tick(track.notes[j].start)
                nextend = pm.time_to_tick(track.notes[j].end)
                # if(start == nextstart or end > nextstart):
                if (math.fabs(start - nextstart) <= 30 or end - nextstart >= 30):
                    pitches.append(track.notes[j].pitch)
                    durations.append((nextend - nextstart) / pm.resolution)
                    j = j + 1
                else:
                    break
            i = j

            if (i < noteLength):
                dic, g = self.rememberANote(musicName, composerName, genreName, trackIndex, pitches[0], order,
                                            durations[0], True)
                str1 = str(order) + ":("
                for t in range(len(pitches)):
                    str1 += str(pitches[t]) + "," + str(durations[t]) + ";"
                #print(str1 + ")")
                order = order + 1
                if (configs.RunTimeState == 1):
                    jstr = json.dumps(g)
                    self.conn.send('/Queue/SampleQueue', jstr)
                    nlist = dic.get('MSMSpike')
                    ns = []
                    for l in nlist:
                        n = l.get('Index')
                        ns.append(n)
                    r_notes.append(ns)
                    tlist = dic.get('MSMTSpike')
                    ts = []
                    for l in tlist:
                        t = l.get('Index')
                        ts.append(t * 60)
                    r_intervals.append(ts)
        return total_dic

    def rememberNotes(self, MusicName, notes, intervals, tempo=True):
        jStr = ''
        # print(intervals)
        notesStr = notes.split(",")
        intervalsStr = intervals.split(",")
        intervaltimes = []
        for i in range(len(intervalsStr) - 1):
            intervaltimes.append(int(intervalsStr[i]))
        print(intervaltimes)
        for i, note in enumerate(notesStr):
            note = int(note)
            if (i < len(notesStr) - 1):
                tinterval = intervalsStr[i]
                tinterval = int(intervalsStr[i])
            self.rememberANote(MusicName, note, i + 1, tinterval, tempo)
        return jStr

    def rememberANote(self, MusicName, ComposerName, genreName, TrackIndex, NoteIndex, order, tinterval, tempo=False):
        if (tempo == False):
            dic = self.cortex.rememberANote(MusicName, NoteIndex, order)
            jsonStr = json.dumps(dic)
            return jsonStr
        else:
            dic, g = self.cortex.rememberANoteandTempo(MusicName, ComposerName, genreName, TrackIndex, NoteIndex, order,
                                                       tinterval)
            return dic, g

    def memorizing(self,MusicName, ComposerName, noteLength, fileName):
        '''
        :param musicName: the name of the piece of music
        :param composerName: the composer who writes this melody
        :param noteLength: the number of notes to be trained(integer), if you want to learn all the notes of a musical work, the value should be specified as "ALL"
        :param fileName: the path and the name of this midi file
        :return: none
        '''
        self.rememberMusic(MusicName, ComposerName)
        self.rememberMIDIMusic(MusicName,ComposerName,noteLength, fileName)

    def recallMusic(self, musicName):
        print("Recall the " + musicName + " ......")
        musicName = musicName.title()
        result = self.cortex.recallMusicPFC(musicName)
        #print(result)
        noteResult = {}
        for tindex,track in result.items():
            ns = track.get('N')
            ts = track.get('T')
            tmp = []
            for key in ns.keys():
                dic = {}
                dic['N']=ns.get(key)
                dic['T']=ts.get(key)
                tmp.append(dic)
            noteResult[tindex] = tmp
        self.writeMidiFile(musicName+"_recall",noteResult)
        print("Recall " + musicName + " finished!")
        return noteResult


    def generateEx_Nihilo(self, firstNote, durations, length,gen_fName):
        '''
        parameters:
        fistNote: Specify the beginning notes to generate a note
        durations: Specify the duration of the beginning notes
        length: the length of the generated music, less than 50 notes
        '''
        print("Generate melody with no style............")
        result = self.cortex.generateEx_Nihilo2(firstNote, durations, length)
        self.writeMidiFile(gen_fName,result)
        print("Generating finished!")
        return result

    def generateEx_NihiloAccordingToGenre(self, genreName, firstNote, durations, length,gen_fName):
        '''
        parameters:
        genreName:Specify the style of genre of the generated melody, for example: Baroque,Classical,Romantic
        fistNote: Specify the beginning notes to generate a note
        durations: Specify the duration of the beginning notes
        length: the length of the generated music, less than 50 notes
        '''
        print("Generate melody with "+ genreName+"\'s style............")
        result = self.cortex.generateEx_NihiloAccordingToGenre(genreName, firstNote, durations, length)
        self.writeMidiFile(gen_fName,result)
        print("Generating finished!")
        return result

    def generateEx_NihiloAccordingToComposer(self, composerName, firstNote, durations, length,gen_fName):
        '''
        parameters:
        composerName:Specify the style of composer of the generated melody, for example: Bach, Mozart and etc.
        fistNote: Specify the beginning notes to generate a note
        durations: Specify the duration of the beginning notes
        length: the length of the generated music, less than 50 notes
        '''
        print("Generate melody with " + composerName + "'s style............")
        result = self.cortex.generateEx_NihiloAccordingToComposer(composerName, firstNote, durations, length)
        self.writeMidiFile(gen_fName,result)
        print("Generating finished!")
        return result

    def generate2TrackMusic(self, firstNotes, durations, lengths):
        result = self.cortex.generate2TrackMusic(firstNotes, durations, lengths)
        return result

    def generateMelodyWithKey(self,tone, firstNotes,durations = None,length = 8):

        result = self.cortex.generateMelodyWithKey(tone, firstNotes,durations,length)
        return result


    def writeMidiFile(self,fileName, mudic):
        '''
            mudic format description:
            mudic = {1:[{'N':71,'T':0.5}.....],
                         2:[{'N':60,'T':0.25}.....],
                         ....
            }
        '''
        fileName += ".mid"
        pm = pretty_midi.PrettyMIDI()
        # Create an Instrument instance for a cello instrument
        for values in mudic.values():

            piano = pretty_midi.Instrument(program=0)
            # Iterate over note names, which will be converted to note number later
            start = 0
            end = 0
            for i, n in enumerate(values):
                # Retrieve the MIDI note number for this note name
                # note_number = pretty_midi.note_name_to_number(note_name)
                # Create a Note instance, starting at 0s and ending at .5s
                end = start + n.get('T')
                note_name = n.get('N')
                if (note_name == -1):
                    note = pretty_midi.Note(
                        velocity=0, pitch=0, start=start, end=end)
                else:
                    note = pretty_midi.Note(
                        velocity=100, pitch=note_name, start=start, end=end)
                # Add it to our cello instrument
                piano.notes.append(note)
                start = end
            # Add the cello instrument to the PrettyMIDI object
            pm.instruments.append(piano)
        # Write out the MIDI data
        pm.write(fileName)