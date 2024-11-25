'''
Created on 2016.5.24

@author: liangqian
'''


import json
import os

def readjsonFile(filename):
    #print(os.path.abspath(os.curdir))
    f = open(filename,'r')
    jsonstrs = f.read()
    #print(jsonstrs)
    jdata = json.loads(jsonstrs)
    return jdata
    
    

#readjsonFile('../jsondata.txt')