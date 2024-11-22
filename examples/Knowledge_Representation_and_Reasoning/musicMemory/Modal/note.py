'''
Created on 2016.7.6

@author: liangqian
'''
from Modal.pitch import Pitch
class Note():
    '''
    Because a chord consist of more than two pitches at the same time, so using
    arrays to record the chord
    '''
    def __init__(self):
        self.pitches = []
#         self.startTime = []
#         self.endTime = []
        self.lastTime = []
       
        