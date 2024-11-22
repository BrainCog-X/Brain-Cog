'''
Created on 2016.6.8

@author: liangqian
'''
import time
import sys
import stomp
class MyListener(object):
    def on_error(self, headers, message):
        print('received an error : %s' % message)
    def on_message(self, headers, message):
        print('%s' % message)

conn = stomp.Connection([('159.226.19.16',61613)])
#conn = stomp.Connection([('10.10.10.106',61613)])   
conn.set_listener('', MyListener())
conn.start()
print('hh')

conn.connect(wait=True,headers={'client-id':'LXYNB','non_persistent':'true'})
 
conn.subscribe(destination='/topic/TEST2.FOO',id='LX', ack='auto',headers={'activemq.subscriptionName':'LXYNB'})
#conn.send(body='hello,garfield!', destination='/topic/myTopic.messages')
 
while(True):
    pass
conn.disconnect()

