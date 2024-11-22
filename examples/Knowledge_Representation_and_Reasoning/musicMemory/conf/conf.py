import numpy
import numpy as np
import pandas as pd
class Conf():
    '''
    classdocs
    '''

    def __init__(self, neutype="LIF", task="MusicLearning", dt=0.1):
        '''
        Constructor
        '''
        self.neuron_type = neutype
        self.task = task
        self.dt = dt
        self.notesMap = {}
        self.GenreMap = {}
        self.emoMap = {}
        self.key_matrix = []
        self.keysMap = {}
        self.index2key = {}
        self.index2mode = {}
        self.keyIndexMap = {}
        self.keyscales = {}
        self.chordsMap = {}
        self.chordsMatrix = np.zeros((7, 7))
        self.RunTimeState = 0  # 0: GUI, 1: Bigdata experiments 2: other

    def readNoteFiles(self):
        # f = open("./Data.txt","r")
        f = open("../inputs/MIDIData.txt", "r")
        while (True):
            line = f.readline()
            if not line:
                break
            else:
                strs = line.split(":")
                index = int(strs[0])
                self.notesMap[index] = strs[1].strip()
        f.close()

    def readGenreFils(self):
        f = open("../inputs/GenreData.txt", "r")
        while (True):
            line = f.readline()
            if not line:
                break
            else:
                strs = line.split(":")
                g = strs[0].strip()
                ns = strs[1].split(",")
                for n in ns:
                    self.GenreMap[(n.strip()).title()] = g.title()
        f.close()

    def readEmotionFiles(self):
        f = open("../inputs/information.csv", "r")
        while (True):
            line = f.readline()
            if not line:
                break
            else:
                strs = line.split(",")
                mn = strs[0].strip()
                e = strs[3].strip()
                self.emoMap[mn.title()] = e.title()
        f.close()

    def readKeysFile(self):
        f = open("../inputs/keyIndex.csv", "r")
        while (True):
            line = (f.readline()).strip()
            if not line:
                break
            else:
                strs = line.split(",")
                toneName = strs[0].strip()
                self.keysMap[toneName] = int(strs[1].strip())
        # print(self.keysMap)
        self.index2key = dict(zip(self.keysMap.values(), self.keysMap.keys()))
        # print(self.index2key)
        self.key_matrix = np.array(pd.read_excel("../inputs/keys.xlsx", sheet_name='keys'))
        self.keyscales = {0: np.array(pd.read_excel("../inputs/keys.xlsx", sheet_name='major')),
                          1: np.array(pd.read_excel("../inputs/keys.xlsx", sheet_name='minor'))}

    def readKeys2IndexFile(self):
        f = open("../inputs/keyIndex.csv", "r")
        while (True):
            line = f.readline().strip()
            if not line:
                break
            else:
                strs = line.split(",")
                self.keyIndexMap[strs[0].strip()] = int(strs[1].strip())
        # print(self.keyIndexMap)

    def readChordsFile(self):
        tmp = pd.read_excel("../inputs/chords.xlsx", sheet_name=None)
        for key, chords in tmp.items():
            chords = np.array(chords)
            self.chordsMap[key.strip()] = chords
        # print(self.chordsMap)
        # 暂时先连接主，下属，属和弦
        self.chordsMatrix = np.array([[1, 0, 0, 1, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [1, 0, 0, 1, 1, 0, 0],
                                      [1, 0, 0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, 0]])

    def readModesFile(self):
        f = open("../inputs/modeindex.csv", "r")
        while (True):
            line = (f.readline()).strip()
            if not line:
                break
            else:
                strs = line.split(",")
                self.index2mode[int(strs[0].strip())] = strs[1].strip()


configs = Conf(neutype = 'Izhikevich')
configs.readNoteFiles()
configs.readGenreFils()
configs.readEmotionFiles()
configs.readKeysFile()
configs.readKeys2IndexFile()
configs.readChordsFile()
configs.readModesFile()