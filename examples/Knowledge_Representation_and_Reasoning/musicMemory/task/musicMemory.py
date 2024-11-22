import sys
sys.path.append("../")
sys.path.append("../../../../")
from api.music_engine_api import EngineAPI
import os


if __name__=="__main__":
    musicEngine = EngineAPI()
    musicEngine.cortexInit()


    input_path = "../testData/"
    #--------------------learning process---------------#

    for composerName in os.listdir(input_path):
        dpath = os.path.join(input_path,composerName)
        if os.path.isdir(dpath):
            for musicName in os.listdir(dpath):
                fileName = (os.path.join(dpath,musicName))
                #Here is the training function, the first and the second parameters refer to the title and composer of a melody.
                #The third parameter indicates the number of notes you want to learn. If you want to learn all the notes of melodies, this value is "ALL".
                musicEngine.memorizing(musicName, composerName, 20, fileName)

    # recall the music based on the name of a music
    musicEngine.recallMusic("Sonate C Major.Mid")


