import os
import sys
import music21 as m21
import numpy as np
firstnotes = np.array([[m21.pitch.Pitch('b-4').midi],
                       [m21.pitch.Pitch('d4').midi],
                       [m21.pitch.Pitch('f3').midi],
                       [m21.pitch.Pitch('B2').midi]])
print(firstnotes)
input_path = "../xmlfiles/hamony dataset/"
for ch in os.listdir(input_path):
    if ch == '.DS_Store': continue
    fileName = os.path.join(input_path, ch)
    for fName in os.listdir(fileName):
        if fName == 'four':
            if fName == '.DS_Store': continue
            fn = os.path.join(fileName+'/four')
            for f in os.listdir(fn):
                if f == '.DS_Store': continue
                print(f)
                musicName = f.split("_")[0]
                fTone = (f.split("_")[1]).split(".")[0]
                print(musicName)
                print(fTone)
                print('---')
