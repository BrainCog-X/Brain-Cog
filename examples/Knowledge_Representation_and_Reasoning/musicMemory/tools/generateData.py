'''
Created on 2016.4.27

@author: liangqian
'''
import json
import random

class joint():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
    def self2dic(self):
        return {'x':self.x,
                'y':self.y,
                'z':self.z}
        
class data():
    
    def __init__(self):
        self.T = 0
        self.position = 0
        self.joints = {}
        
    def self2dic(self):
        return {'T':self.T,
                'position':self.position
                }

Data_List = []

# for i in range(1,419):
#     d = data()
#     d.T = i
#     d.position = random.randint(0,100)
#     dic = d.self2dic()
#     Data_List.append(d)
# 
# dic = {}
# dic['Data'] = Data_List
#     
# # using json to write file
# strs = (json.dumps(dic))
# 
# fout = open('position.txt','w')
# fout.write(strs)
# fout.close()
# 
# 
# fin = open('position.txt','r')
# data_json = fin.read()
# 
# print(data_json)
f = open('../Data.txt','r')
line = f.readline()
while(len(line) != 0):
    if('T=' in line):
        ss = line.split('=')
        d = data()
        d.T = int(ss[1].strip())
        d.position = random.randint(0,100)
        line = f.readline()
        while(len(line.strip()) != 0):
            sss = line.split(' ')
            jj = joint()
            s = sss[1].split('=')
            jj.x = float(s[1].strip())
            s = sss[2].split('=')
            jj.y = float(s[1].strip())
            s = sss[3].split('=')
            jj.z = float(s[1].strip())
            d.joints[sss[0]] = jj
            line = f.readline()
        Data_List.append(d)
        print(len(d.joints))
    line = f.readline()     
print(len(Data_List))        
