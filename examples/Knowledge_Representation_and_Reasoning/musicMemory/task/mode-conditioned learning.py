import sys
import os
import time
sys.path.append("../../../../")
sys.path.append("../")
import numpy as np
import music21 as m21
from conf.conf import *
from api.music_engine_api import EngineAPI


if __name__=="__main__":
    musicEngine = EngineAPI()
    musicEngine.cortexInit()
    #------------Bach dataset learning----------------#
    paths = m21.corpus.getComposer('bach')
    print(len(paths))
    for path in paths:
        musicName = (str(path).split('\\'))[-1]
        print(musicName)
        if musicName.split('.')[-1] != 'mxl': continue
        xmldata = m21.corpus.parse(path)
        musicEngine.rememberMusic(musicName, "None")
        musicEngine.learnFourPartMusic(xmldata, musicName, "None")

    #------------generation test----------------#
    key = 'C major'
    firstnotes = np.array([[m21.pitch.Pitch('E5').midi],
                           [m21.pitch.Pitch('G4').midi],
                           [m21.pitch.Pitch('C4').midi],
                           [m21.pitch.Pitch('C3').midi]])

    result = musicEngine.generateMelodyWithKey(configs.keyIndexMap.get(key),firstnotes,None,4)
    steam1 = m21.stream.Stream()
    for i,part in result.items():
        pt = m21.stream.Stream()
        for v in part:
            p = v.get("N")
            d = v.get("T")
            n = m21.note.Note(p)
            n.quarterLength = d
            pt.append(n)
        steam1.insert(0,pt)
    opath = '../result_output/tone learning/'
    nowtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    t2 = ''.join([x for x in nowtime if x.isdigit()])
    steam1.write('midi', fp=opath+key+"_"+t2+'.mid')
