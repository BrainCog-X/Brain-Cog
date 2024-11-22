'''
Created on 2016.6.29

@author: liangqian
'''

import pygame,sys
pygame.init()
pygame.mixer.init()
pygame.time.delay(1000)
pygame.mixer.music.load("do.wav")
pygame.mixer.music.play()
while 1:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            sys.exit()
