'''
Created on 2018.9.7

@author: liangqian
'''
import time
import sys
import stomp

def createMSQ():
    queue_name = '/queue/SampleQueue'
    conn = stomp.Connection([('localhost',61613)])
    #conn.start()
    print("building connection to activemq......")
    conn.connect()
    #return conn
#     for i in range (10):
#         msg = 'this is the '+ str(i) + 'th messages'
#         conn.send(queue_name,msg)
#         print(msg)
#     conn.disconnect()
    return conn

#createMSQ()