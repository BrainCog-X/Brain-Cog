import wave
import struct
import os
import numpy as np

f = 440
framerate = 44100.0
fw = wave.open("sine.wav","wb")
fw.setnchannels(1)
fw.setframerate(framerate)
fw.setsampwidth(2)
tt = np.arange(0, 1, 1.0/framerate)

data = [2000*(np.sin(2*np.pi*f*t)+np.sin(2*np.pi*2*f*t)+np.sin(2*np.pi*3*f*t)) for t in tt]
print(data)
for d in data:
    fw.writeframes(struct.pack('h',int(d)))
fw.close()