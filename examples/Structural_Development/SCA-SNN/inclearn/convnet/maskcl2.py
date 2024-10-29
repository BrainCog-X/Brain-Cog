import numpy as np
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
from copy import deepcopy

def unit(x):
    if x.size()[0]>0:
        xnp=x.cpu().numpy()
        maxx=torch.max(x)
        #maxx=np.percentile(xnp, 99.5)
        minx=torch.min(x)
        marge=maxx-minx
        if marge!=0:
            xx=(x-minx)/marge
            xx=torch.clip(xx, 0,1)
        else:
            xx=0.5*torch.ones_like(x)
        return xx
    else:
        return x
        
class Mask:
    def __init__(self, model):
        self.model = model
        self.mat = {}
        self.p_index={}
        self.p_num={}
        self.k=15
        self.task_ready={}
        self.regutask_ready={}
        self.taskmask={}
        self.init_rate=0.3
        self.grow_rate=0.125

        self.prunconv_init=0.8
        self.prunfc_init=1.3
        self.prunconv_grow=0.5
        self.prunfc_grow=1

        self.n_delta={}
        self.ren_delta={}
        self.reduce={}
        self.rereduce={}
        self.taskww={}
        self.tasknore={}

    def init_length(self,task=0,task_nn=None):
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and item.size()[-1]!=1:
                print(index,item.size())
                self.mat[index]=torch.ones(item.size(),device=device)
        for t in range(task):
            self.rereduce[t]={}
            for index, item in enumerate(self.model.parameters()):
                if True:
                    if index<=20:
                        c_index=0
                    elif index<=45:
                        c_index=1
                    elif index<=70:
                        c_index=2
                    else:
                        c_index=3
                taskindb=task_nn[t-1][c_index]
                taskinda=task_nn[t][c_index]
                lenre=taskinda-taskindb
                self.rereduce[t][index] = 1*torch.ones(lenre,device=device)

        return self.mat
            

    def get_filter_reuse(self,index,ww,task,epoch,c_index,cdim_before=None,task_nn=None,all_dist=None,bias=0): 
        lenre=cdim_before[1]-cdim_before[0]
        similar=1-all_dist+bias
        if similar<0.2:
            similar=0.2
        if similar>0.9:
            similar=0.9

        revalue=similar*torch.ones(lenre).cuda() #1/8,1/4,1/2,1,1.5

        if len(ww.size()) == 4:
            # p_www=ww*self.mat[index]
            p_ww=torch.sum(torch.sum(torch.sum(ww,dim=3),dim=2),dim=1)

            # p_ww=p_ww[cdim_before[0]:cdim_before[1]]
            
            ren_delta=-(2*unit(p_ww)-revalue)#revalue0.8
            #print(self.ren_delta[index])
            pos=torch.nonzero(ren_delta>0)
            ren_delta[pos]=ren_delta[pos]+3
            self.rereduce[task][index]=self.rereduce[task][index]*0.999+ren_delta*math.exp(-int((epoch-1)/2))
            p_ind = torch.nonzero(self.rereduce[task][index] <0)
            matkey=self.mat.keys()
            matkey=torch.tensor(list(matkey))
            matindex=torch.nonzero(matkey==index)
            next_index=matkey[matindex+1]
            for x in range(0, len(p_ind)):
                self.mat[next_index.item()][:,p_ind[x]+cdim_before[0]]=0
            b = self.mat[next_index.item()][:,cdim_before[0]:cdim_before[1]].reshape(-1).cpu().numpy()
            pruning=100*(len(b)- np.count_nonzero(b))/len(b)
            #print(index,self.rereduce[task][index].mean(),self.rereduce[task][index].max(),self.rereduce[task][index].min(),len(b)-np.count_nonzero(b),pruning)

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_mask(self,task,epoch,dim_cur=None,task_nn=None,all_dist=None,all_model=None):
        for t in range(task):
            similart=all_dist[t]
            for index, item in enumerate(all_model[t].parameters()):
                if len(item.size()) > 2 and item.size()[-1]!=1 and index<95:
                    if index<=20:
                        c_index=0
                    elif index<=45:
                        c_index=1
                    elif index<=70:
                        c_index=2
                    else:
                        c_index=3
                        
                    if index<=25:
                        bias=0.2
                    elif index<=50:
                        bias=0.1
                    elif index<=74:
                        bias=-0.1
                    else:
                        bias=0.2
                        
                    taskindb=task_nn[t-1][c_index]
                    taskinda=task_nn[t][c_index]
                    cdim_before=[taskindb,taskinda]
                    self.get_filter_reuse(index,abs(item.grad),t,epoch,c_index,cdim_before,task_nn=task_nn,all_dist=similart,bias=bias)

    def do_mask(self,task):
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and item.size()[-1]!=1:
                ww=item.data
                item.data=ww*self.mat[index].cuda()
        return self.mat

    def if_zero(self):
        cc=[]
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and item.size()[-1]!=1 and index>0:
                b = item.data.view(-1).cpu().numpy()
                print("number of weight is %d, zero is %.3f" %(len(b),100*(len(b)- np.count_nonzero(b))/len(b)))
                cc.append(100*(len(b)- np.count_nonzero(b))/len(b))
        return cc

