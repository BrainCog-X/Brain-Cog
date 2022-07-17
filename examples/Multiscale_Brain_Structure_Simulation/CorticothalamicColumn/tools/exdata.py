import os
import csv
#import pandas as pd
class EXDATA():
    def __int__(self):
        pass

    def getCortexData(self):
        neurondic = {}
        f = open("./tools/cortical.csv","r")
        line = f.readline()
        count = 0
        while(True):
            line = (f.readline()).strip()
            if(len(line) <= 0): break;
            strs = line.split(",")
            info = {}
            info['neuronname'] = strs[0].strip()
            info['neunum'] = float(strs[1])
            info['synnum'] = int(strs[2])
            info['area'] = str(strs[3])
            neurondic[strs[0].strip()] = info
            count += 1
        f.close()
        print(neurondic)
        return neurondic

    def getCortexData2(self):
        neurondic = {}
        data = pd.read_csv("../tools/cortical.csv")
        print('debug')


    def getLayerData(self):
        layerdic = {}
        f = open("./tools/layer.csv","r")
        strs = (f.readline()).strip()
        str = strs.split(",")
        count = 0
        while(True):
            info = {}
            line = (f.readline()).strip()
            if(len(line) <= 0):break
            v = line.split(",")
            for i in range(len(str)):
                if(i > 0):
                    v[i] = float(v[i])
                info[str[i]] = v[i]
            layerdic[count] = info
            count += 1
        f.close()
        return layerdic

    def getNeuronData(self):
        neurondic = {}
        f = open("./tools/neuron.csv","r")
        strs = (f.readline()).strip()
        str = strs.split(",")
        while (True):
            info = {}
            line = (f.readline()).strip()
            if (len(line) <= 0): break
            v = line.split(",")
            for i in range(len(str)):
                if(len(v[i]) <= 0): break
                if(i > 4): v[i] = float(v[i])
                info[str[i].strip()] = v[i]
            neurondic[v[0].strip()] = info
        f.close()
        return neurondic

    def getSynapseData(self, postneuron):
        synapsemap = {}
        f = open("./tools/synapse.csv","r")
        fields = (f.readline()).strip()
        fields = fields.split(",")
        while(True):
            line = (f.readline()).strip()
            if(len(line) <= 0):break
            strs = line.split(",")
            info = {}
            for i,v in enumerate(fields):
                if(i > 1): strs[i] = float(strs[i])
                info[v] = strs[i]
            if(synapsemap.get(strs[0]) == None):
                syndic = {}
                syndic[len(syndic)] = info
                synapsemap[strs[0]] = syndic
            else:
                syndic = synapsemap.get(strs[0])
                syndic[len(syndic)] = info
        f.close()

        return synapsemap.get(postneuron)


#tmp = EXDATA()
#tmp.getCortexData()
#tmp.getLayerData()
#tmp.getNeuronData()
#result = tmp.getSynapseData('p4')
#print(result)