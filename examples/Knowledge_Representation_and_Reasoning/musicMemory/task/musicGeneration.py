import sys
sys.path.append("../")
from api.music_engine_api import EngineAPI
import os

if __name__=="__main__":
    #----------------------------------Init------------------------------#
    musicEngine = EngineAPI()
    musicEngine.cortexInit()

    #----------------------------Learning process------------------------#
    input_path = "../testData/"
    for composerName in os.listdir(input_path):
        dpath = os.path.join(input_path,composerName)
        if os.path.isdir(dpath):
            for musicName in os.listdir(dpath):
                fileName = (os.path.join(dpath,musicName))
                musicEngine.memorizing(musicName,composerName,20,fileName)


    #-------------------------Generation Process------------------------#
    beginnotes = {1:[-1,67],
                  2:[-1]}
    begindurs = {1:[0.5,0.25],
                 2:[0.5]}
    lengths = [10,8]

    genreName = "Classical"
    composerName = "Bach"
    #Generate a piece of music melody#
    musicEngine.generateEx_Nihilo(beginnotes.get(2),begindurs.get(2),20,"melody_generated")
    #Generate a piece of melody with a composer style
    musicEngine.generateEx_NihiloAccordingToComposer(composerName,beginnotes.get(2),begindurs.get(2),15,"Bach_generated")
    #Generate a piece of melody with a genre style
    musicEngine.generateEx_NihiloAccordingToGenre(genreName,beginnotes.get(1),begindurs.get(1),15,"Classical_generated")