import numpy as np
import torch
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random

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
        self.taskmask={}
        self.init_rate=0.3
        self.grow_rate=0.125

        self.prunconv_init=0.8
        self.prunfc_init=1.3
        self.prunconv_grow=0.5
        self.prunfc_grow=1

        self.n_delta={}
        self.reduce={}
        self.taskww={}

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1:
                print(index,item.size())
                self.mat[index]=torch.ones(item.size(),device=device)
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1:
                if index<=40:
                    self.p_index[index]=torch.tensor([])
                    self.task_ready[index]=torch.zeros(item.size(),device=device)
                    self.reduce[index] = 5*torch.ones(item.size()[0],device=device)
                    if len(item.size()) == 4:
                        self.p_num[index]=torch.zeros(item.size()[0],device=device)
                        self.mat[index][int(self.init_rate*item.size()[0]):]=0.0
                        if index+5<40:
                            self.mat[index+5][:,int(self.init_rate*item.size()[0]):]=0.0
                        if index+5==40:
                            self.mat[index+5][:,int(self.init_rate*item.size()[0])*16:]=0.0
                    if len(item.size()) == 2:
                        self.p_num[index]=torch.zeros(item.size()[0]*item.size()[1],device=device)
                        self.mat[index][int(self.init_rate*item.size()[0]):]=0.0
                if index>40:
                    self.mat[index]=torch.ones(item.size(),device=device)
                    if index==44:
                        self.mat[index]=torch.zeros(item.size(),device=device)
                        self.mat[index][:,:int(self.init_rate*item.size()[1])]=1.0
        return self.mat
            
    def get_filter_codebook(self,index,ww,task,epoch): 
        if task==0:
            pruncon=self.prunconv_init
            prunfc=self.prunfc_init
        else:
            pruncon=self.prunconv_grow
            prunfc=self.prunfc_grow

        if len(ww.size()) == 4:
            p_ww=ww.view(ww.size()[0],-1)
            p_ww=torch.sum(p_ww,dim=1)

            task_use=self.mat[index]-torch.sign(self.task_ready[index])
            nouse=torch.sum(task_use.view(ww.size()[0],-1),dim=1)
            no=torch.nonzero(nouse<0.1)
            p_ww[no]=p_ww.max()
            
            self.n_delta[index]=(unit(p_ww)*2-pruncon)
            pos=torch.nonzero(self.n_delta[index]>0)
            self.n_delta[index][pos]=self.n_delta[index][pos]+3
            self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch-1)/13))
            p_ind = torch.nonzero(self.reduce[index] <0)
            print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),len(p_ind))
            for x in range(0, len(p_ind)):
                self.mat[index][p_ind[x]] = 0
                if index+5<40:
                    self.mat[index+5][:,p_ind[x]]=0
                if index+5==40:
                    self.mat[index+5][:,p_ind[x]*16:(p_ind[x]+1)*16]=0
            self.mat[index]=torch.sign(self.mat[index]+ self.task_ready[index])

        if len(ww.size()) == 2:
            p_ww=torch.sum(ww,dim=1)

            task_use=self.mat[index]-torch.sign(self.task_ready[index])
            nouse=torch.sum(task_use,dim=1)
            no=torch.nonzero(nouse<0.1)
            p_ww[no]=p_ww.max()

            self.n_delta[index]=(unit(p_ww)*2-prunfc)
            print(self.n_delta[index].mean(),self.n_delta[index].max(),self.n_delta[index].min())
            pos=torch.nonzero(self.n_delta[index]>0)
            self.n_delta[index][pos]=self.n_delta[index][pos]+3
            self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch-1)/13))
            p_ind = torch.nonzero(self.reduce[index] <0)
            print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),len(p_ind))
            index_ta=44+task*2
            for x in range(0, len(p_ind)):
                self.mat[index][p_ind[x]] = 0
                self.mat[index_ta][:,p_ind[x]]=0
            self.mat[index]=torch.sign(self.mat[index]+self.task_ready[index])

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_mask(self,task,epoch):
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and index<=40:
                self.get_filter_codebook(index,abs(item.data),task,epoch)

    def do_mask(self,task):
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and index<=40:
                ww=item.data
                item.data=ww*self.mat[index].cuda()
        return self.mat

    def init_grow(self,task):
        self.taskmask[task]={}
        index_ta=44+task*2
        self.mat[index_ta]=torch.zeros(self.mat[index_ta].size(),device=device)
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and index<=40:
                self.task_ready[index]=self.task_ready[index]+self.mat[index]
                self.taskmask[task][index]=self.mat[index]
                self.taskww[index]=item.data.clone()
                ind_all=[x for x in range(item.size()[0])]
                pp=list(np.array(self.p_index[index]))
                ind_empty=set(ind_all)-set(pp)
                ind_empty=list(ind_empty)
                # random.shuffle(ind_empty)
                ind_grow=ind_empty[:int(item.size()[0]*self.grow_rate)]
                ind_grow=torch.tensor(ind_grow)
                ww_g=torch.empty(item.size(),device=device)
                if index<40:
                    torch.nn.init.kaiming_uniform_(ww_g, a=math.sqrt(5))
                if index==40:
                    kk=1/math.sqrt(item.size()[1])
                    torch.nn.init.uniform_(ww_g,a=-kk,b=kk)
                for x in range(0, len(ind_grow)):
                    self.mat[index][ind_grow[x]] = 1.0
                    item.data[ind_grow[x]]=ww_g[ind_grow[x]]
                    if index==40:
                        self.mat[index_ta][:,ind_grow[x]]=1.0
                self.mat[index]=torch.sign(self.mat[index]+self.task_ready[index])
                self.p_num[index]=torch.zeros(item.size()[0],device=device)
                self.reduce[index] = 5*torch.ones(item.size()[0],device=device)
                #self.mat[index_ta]=self.mat[index_ta]+self.mat[index_ta-2]
        # nn=torch.sum(self.task_ready[24],dim=1)
        # use_nn=torch.nonzero(nn>1)
        # for x in range(0, len(use_nn)):
        #     self.mat[index_ta][:,use_nn[x]] = 1.0
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and index<40:
                nsum=self.mat[index].view(item.size()[0],-1)
                nn=torch.sum(abs(nsum),dim=1)
                empy_nn=torch.nonzero(nn<0.00001)
                for x in range(0, len(empy_nn)):
                    if index+5<40:
                        self.mat[index+5][:,empy_nn[x]]=0
                    if index+5==40:
                        self.mat[index+5][:,empy_nn[x]*16:(empy_nn[x]+1)*16]=0

        return self.mat,self.task_ready,self.taskmask,self.taskww

    def if_zero(self):
        cc=[]
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and index<=40:
                b = item.data.view(-1).cpu().numpy()
                print("number of weight is %d, zero is %.3f" %(len(b),100*(len(b)- np.count_nonzero(b))/len(b)))
                cc.append(100*(len(b)- np.count_nonzero(b))/len(b))
                if index==40:
                    nouse=torch.sum(self.mat[index],dim=1)
                    no=torch.nonzero(nouse<0.1)
                    print(len(no))
                    cc.append(len(no))
        return cc

    def record(self):
        for index, item in enumerate(self.model.parameters()):
            if len(item.size()) > 1 and index<=40:
                nsum=self.mat[index].view(item.size()[0],-1)
                nn=torch.sum(abs(nsum),dim=1)
                epoch_select=torch.nonzero(nn>0.00001)
                select=set(epoch_select)|set(self.p_index[index])
                self.p_index[index]= torch.tensor(list(select))
        return self.p_index