'''
Created on 2016.7.7

@author: liangqian
'''
from Modal.note import Note
from Modal.cluster import Cluster
from conf.conf import configs
from Modal.pitch import Pitch
class APAC():
    '''
    anterior primary auditory cortex,encoding the musical notes
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.notes = []
        #self.cluster = Cluster()
        
    def encodingNote(self,NoteID):
        NoteName = configs.notesMap.get(int(NoteID))
        n = Pitch()
        n.name = NoteName
        n.frequence = int(NoteID)
        self.notes.append(n)
        return n
    
    def encodingMIDINote(self,p):
        NoteName = configs.notesMap.get(p.frequence)
        p.name = NoteName
    