import numpy as np
import torch
import math
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mask:
    def __init__(self, model,count_thre):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.distance_rate = {}
        self.filter_small_index = {}
        self.filter_large_index = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

        # dendritic dynamics
        self.cur_range_pos = {}  # current range foe every weight
        self.cur_range_neg = {}  # current range foe every weight
        self.dsum_pos_out = {}  # current range foe every weight
        self.dsum_neg_out = {}  # current range foe every weight
        self.dsum_pos_in = {}  # current range foe every weight
        self.dsum_neg_in = {}  # current range foe every weight
        self.dendritic_previous_pos = {}  # the index for beyond range weight
        self.dendritic_previous_neg = {}  # the index for within range weight
        self.dendritic_previous_in = {}  # the index for within range weight
        self.dendritic_count_pos = {}  # the count of beyond range for every weight
        self.dendritic_count_neg = {}  # the count of within range for every weight
        self.dendritic_count_in = {}  # the count of within range for every weight
        self.out = 0
        self.count_thre=count_thre
        self.weight_previous = {}
        self.mask={}
        self.codebook = {}
        self.codebookww={}
        self.his_ind={}
        self.his_groth={}
        self.his_gro_count={}
        self.prune={}
        self.pruncc={}
        self.prunn={}
        self.his_prun={}
        self.convlayer =model.convlayer 
        for index in self.convlayer:
            if index<self.convlayer[-1]:
                self.pruncc[index]=0.5
            else:
                self.prune[index]=2
            self.prunn[index]=0
        self.feature=self.model.feature
        self.fc1=self.model.fc_prun[0]


    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index in self.convlayer:
            if index<self.convlayer[-1]:
                ww=self.feature[index].conv.weight
            if index==self.convlayer[-1]:
                ww=self.fc1.fc.weight
            self.model_size[index]=ww.size()
            self.codebook[index]=torch.ones_like(ww)

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
            # dendritic parameters initialize
            self.his_groth[index1]=np.array([self.model_length[index1]+1])
            self.his_prun[index1]=np.array([self.model_length[index1]+1])
            self.his_ind[index1]=np.array([-1])
            self.dendritic_previous_pos[index1] = np.array([self.model_length[index1] + 1]) 
            self.dendritic_previous_neg[index1] = np.array([self.model_length[index1] + 1])
            self.dendritic_count_pos[index1] = np.zeros((self.model_length[index1],))
            self.dendritic_count_neg[index1] = np.zeros((self.model_length[index1],))
            self.dendritic_count_in[index1] = np.zeros(
                (self.model_length[index1],))  # the count of within range for every weight
            self.dendritic_previous_in[index1] = np.array(
                [self.model_length[index1] + 1])  # the index for within range weight
            self.dsum_pos_out[index1] = np.zeros((self.model_length[index1],))  # current range for every weight
            self.dsum_neg_out[index1] = np.zeros((self.model_length[index1],))  # current range for every weight
            self.dsum_pos_in[index1] = np.zeros((self.model_length[index1],))  # current range for every weight
            self.dsum_neg_in[index1] = np.zeros((self.model_length[index1],))  # current range for every weight
            if index1<self.convlayer[-1]:
                self.his_gro_count[index1] = np.zeros((self.model_size[index1][0]*self.model_size[index1][1]))
            else:
                self.his_gro_count[index1] = np.zeros((self.model_length[index1],))

        for index in self.convlayer:
            if index<self.convlayer[-1]:
                ww=self.feature[index].conv.weight
            if index==self.convlayer[-1]:
                ww=self.fc1.fc.weight
            weight_tmp = ww.data.view(-1)  # one conv one weight vector
            weight_tmp_np = weight_tmp.cpu().numpy()
            self.weight_previous[index] = abs(weight_tmp_np)
            self.cur_range_pos[index] = weight_tmp_np.max() * np.ones((self.model_length[index],))  # args.init_range
            self.cur_range_neg[index] = -1 * weight_tmp_np.max() * np.ones((self.model_length[index],))  # -1*args.init_range


    def init_mask_dsd(self):
        for index in self.convlayer:
            if index<self.convlayer[-1]:
                ww=self.feature[index].conv.weight
            if index==self.convlayer[-1]:
                ww=self.fc1.fc.weight
                ww=ww.data*self.codebook[index].cuda()  
                self.cur_range_pos[index], self.cur_range_neg[index] = \
                    self.get_range_weight(ww,self.model_length[index],index,self.count_thre)  # update current rang
    
    def get_range_weight(self, weight_torch, length, i, count_thre):
        # >r+
        weight_vec = weight_torch.view(-1)  # one conv one weight vector
        weight_vec_np = weight_vec.cpu().numpy()
        weight_np_abs = abs(weight_vec_np)
        dendritic_pos_tmp = np.where((weight_vec_np >= self.cur_range_pos[i]))  # find the weight beyong range
        pos_index = set(dendritic_pos_tmp[0]) & set(self.dendritic_previous_pos[i])  # calculate intersection  consectively
        pos_zero = set([i for i in range(length)]) - pos_index  # non-intersection weight will be count from 0
        pos_index = np.array(list(pos_index))
        pos_zero = np.array(list(pos_zero))
        if pos_zero.size > 0:
            self.dendritic_count_pos[i][pos_zero] = 0
            self.dsum_pos_out[i][pos_zero] = 0
        if pos_index.size > 0:
            self.dendritic_count_pos[i][pos_index] = self.dendritic_count_pos[i][pos_index] + 1  # intewrsection +1
            self.dsum_pos_out[i][pos_index] += weight_vec_np[pos_index] - self.cur_range_pos[i][pos_index]
        dendritic_index = np.where(self.dendritic_count_pos[i] >= count_thre)  # count>threshold
        self.out = self.out + len(dendritic_index[0])
        self.cur_range_pos[i][dendritic_index]+=(self.dsum_pos_out[i][dendritic_index] / count_thre)  #  self.cur_range_pos[i][dendritic_index]+0.025 # 
        self.dendritic_count_pos[i][dendritic_index] = 0  # intrsection count set to 0
        self.dsum_pos_out[i][dendritic_index] = 0
        self.dendritic_previous_pos[i] = dendritic_pos_tmp[0]  # update previous
        # <r-
        dendritic_neg_tmp = np.where((weight_vec_np <= self.cur_range_neg[i]))  # find the weight beyong range
        neg_index = set(dendritic_neg_tmp[0]) & set(
            self.dendritic_previous_neg[i])  # calculate intersection  consectively
        neg_zero = set([i for i in range(length)]) - neg_index  # non-intersection weight will be count from 0
        neg_index = np.array(list(neg_index))
        neg_zero = np.array(list(neg_zero))
        if neg_zero.size > 0:
            self.dendritic_count_neg[i][neg_zero] = 0
            self.dsum_neg_out[i][neg_zero] = 0
        if neg_index.size > 0:
            self.dendritic_count_neg[i][neg_index] = self.dendritic_count_neg[i][neg_index] + 1  # intewrsection +1
            self.dsum_neg_out[i][neg_index] += weight_vec_np[neg_index] - self.cur_range_neg[i][neg_index]
        dendritic_index_neg = np.where(self.dendritic_count_neg[i] >= count_thre)  # count>threshold
        self.out = self.out + len(dendritic_index_neg[0])
        self.cur_range_neg[i][dendritic_index_neg] += (self.dsum_neg_out[i][dendritic_index_neg] / count_thre)  #self.cur_range_neg[i][dendritic_index_neg]-0.025 #
        self.dendritic_count_neg[i][dendritic_index_neg] = 0  # intrsection count set to 0
        self.dsum_neg_out[i][dendritic_index_neg] = 0
        self.dendritic_previous_neg[i] = dendritic_neg_tmp[0]  # update previous
        # r-~r+
        dendritic_in_tmp = np.where((weight_np_abs < self.weight_previous[i]))  # find the weight beyong range
        in_index = set(dendritic_in_tmp[0]) & set(self.dendritic_previous_in[i])  # calculate intersection  consectively
        in_zero = set([i for i in range(length)]) - in_index  # non-intersection weight will be count from 0
        in_index = np.array(list(in_index))
        in_zero = np.array(list(in_zero))
        if in_zero.size > 0:
            self.dendritic_count_in[i][in_zero] = 0
            self.dsum_neg_in[i][in_zero] = 0
        if in_index.size > 0:
            self.dendritic_count_in[i][in_index] = self.dendritic_count_in[i][in_index] + 1  # intewrsection +1
            self.dsum_neg_in[i][in_index] += weight_np_abs[in_index] - self.weight_previous[i][in_index]
        dendritic_index_in = np.where(self.dendritic_count_in[i] >= count_thre)  # count>threshold
        self.cur_range_pos[i][dendritic_index_in] =  0.75*self.cur_range_pos[i][dendritic_index_in]  # self.cur_range_pos[i][dendritic_index_in]-0.025
        self.cur_range_neg[i][dendritic_index_in] =  0.75*self.cur_range_neg[i][dendritic_index_in]  # self.cur_range_neg[i][dendritic_index_in]+0.025
        self.dendritic_count_in[i][dendritic_index_in] = 0  # intrsection count set to 0
        self.dsum_neg_in[i][dendritic_index_in] = 0
        self.dendritic_previous_in[i] = dendritic_in_tmp[0]  # update previous
        self.weight_previous[i] = weight_np_abs
        print('dendritic dynamics done', np.mean(self.cur_range_pos[i]), np.mean(self.cur_range_neg[i][0]))
        return self.cur_range_pos[i], self.cur_range_neg[i]

    def do_mask_dsd(self):
        for index in self.convlayer:
            if index<self.convlayer[-1]:
                ww=self.feature[index].conv.weight
            if index==self.convlayer[-1]:
                ww=self.fc1.fc.weight
            a = ww.data.view(self.model_length[index])
            a = a.cpu().numpy()
            a[a > self.cur_range_pos[index]] = self.cur_range_pos[index][
                a > self.cur_range_pos[index]]  # weight beyond range set to range
            a[a < self.cur_range_neg[index]] = self.cur_range_neg[index][a < self.cur_range_neg[index]]
            a = torch.FloatTensor(a).cuda()
            ww.data = a.view(self.model_size[index])
        print("mask Done")

    def do_pruning_dsd(self,epoch):
        for index in self.convlayer:
            if index<self.convlayer[-1]:
                rate = int(self.model_size[index][0]* self.pruncc[index]/100)
                b=unit(self.cur_range_pos[index]+abs(self.cur_range_neg[index]))
                b=b*self.codebook[index].view(-1).cpu().numpy()
                a=torch.tensor(b)
                prer = a.reshape(self.model_size[index][0], -1)
                presum = torch.sum(prer, dim=1)
                n_range =presum
                n_index = torch.argsort(n_range)
                n_index=n_index[0:rate]
                n_index=n_index.cpu().numpy()
                ind=set(n_index)-set(self.his_prun[index])
                ind=np.array(list(ind))
                self.codebook[index][ind] = 0
                ww=self.feature[index].conv.weight
                a = ww.data*self.codebook[index].cuda()
                ww.data = a
                self.his_prun[index]=set(self.his_prun[index]) | set(ind)
                self.cal_prune(index,epoch)
            if index==self.convlayer[-1]:
                rate = int(self.model_size[index][0] * self.prune[index]/100)
                b=unit(self.cur_range_pos[index]+abs(self.cur_range_neg[index]))
                b=b*self.codebook[index].view(-1).cpu().numpy()
                a=torch.tensor(b)
                prer = a.reshape(self.model_size[index][0], -1)
                presum = torch.sum(prer, dim=1)
                n_range =presum
                n_index = torch.argsort(n_range)[0:rate]
                n_index=n_index.cpu().numpy()
                ind=set(n_index)-set(self.his_prun[index])
                ind=np.array(list(ind))
                self.codebook[index][ind] = 0
                ww=self.fc1.fc.weight
                a = ww.data*self.codebook[index].cuda()
                ww.data = a
                self.his_prun[index]=set(self.his_prun[index]) | set(ind)
                self.cal_prune(index,epoch)

    def cal_prune(self,index,epoch):
        if index<self.convlayer[-1]:
            ind=self.convlayer.index(index)
            code=self.codebook[self.convlayer[ind+1]].view(self.model_size[self.convlayer[ind+1]][0],-1)
            sumbook=torch.sum(code,dim=1)
            prun=torch.where(sumbook<20)[0]
            prunn=prun.size()[0]
            codei=self.codebook[index].view(self.model_size[index][0],-1)
            sumbooki=torch.sum(codei,dim=1)
            pruni=torch.where(sumbooki<20)[0]
            prunni=pruni.size()[0]
            self.pruncc[index]=self.pruncc[index]+math.exp(-(epoch-2*self.count_thre-1)/5)*(self.model_size[index][0]-prunni)/(self.model_size[self.convlayer[ind+1]][0]-prunn)
            print(prunn,self.pruncc[index])
        else:
            if epoch<=60:
                aphla=0.5*math.exp(-(epoch-37)) #0.25**(epoch-37) math.exp(-(epoch-37))
            if epoch>60:
                aphla=0.0005
            sumbook=torch.sum(self.codebook[self.convlayer[-1]],dim=1)
            prun=torch.where(sumbook<50)[0]
            prunn=prun.size()[0]
            self.prune[index]=self.prune[index]+aphla*(512-prunn)/10
            print(prunn,self.prune[index])

    def do_growth_ww(self,epoch):
        for index in self.convlayer:
            if index<self.convlayer[-1]:
                ww=self.feature[index].conv.weight
                ww=ww.data
                ww=torch.sum(torch.sum(ww,dim=2),dim=2).view(-1).cpu().numpy()
                code=torch.sum(torch.sum(self.codebook[index],dim=2),dim=2)
                p_index=np.where(code.view(-1).cpu().numpy()==0)[0]
                rate=65+1.1**(epoch- 2*self.count_thre-1)
                if rate>99:
                    rate=99
                gg=np.percentile(ww, rate)
                grow=np.where(ww>gg)[0]
                growth_ind=set(grow) & set(p_index)
                growth_index=growth_ind & set(self.his_groth[index])
                zero_index=set([i for i in range(ww.size)]) - growth_index
                growth_index=np.array(list(growth_index))
                zero_index=np.array(list(zero_index))
                if zero_index.size>0:
                    self.his_gro_count[index][zero_index]=0
                if growth_index.size>0:
                    self.his_gro_count[index][growth_index]=self.his_gro_count[index][growth_index]+1
                gr_index=np.where(self.his_gro_count[index]> self.count_thre)[0]
                self.codebook[index]=self.codebook[index].view(-1)
                for x in range(len(gr_index)):
                    self.codebook[index][gr_index[x]*9:(gr_index[x]+1)*9]=1
                self.codebook[index]=self.codebook[index].view(self.model_size[index])
                print(len(gr_index),len(growth_ind),len(p_index))
                self.his_groth[index]=growth_ind
                self.his_gro_count[index][gr_index]=0
            if index==self.convlayer[-1]:
                ww=self.fc1.fc.weight
                ww=ww.data
                ww=ww.view(-1).cpu().numpy()
                p_index=np.where(self.codebook[index].view(-1).cpu().numpy()==0)[0]
                rate=60+1.1**(epoch- 2*self.count_thre-1)
                if rate>99:
                    rate=99
                gg=np.percentile(ww, rate)
                grow=np.where(ww>gg)[0]
                growth_ind=set(grow) & set(p_index)
                growth_index=growth_ind & set(self.his_groth[index])
                zero_index=set([i for i in range(ww.size)]) - growth_index
                growth_index=np.array(list(growth_index))
                zero_index=np.array(list(zero_index))
                if zero_index.size>0:
                    self.his_gro_count[index][zero_index]=0
                if growth_index.size>0:
                    self.his_gro_count[index][growth_index]=self.his_gro_count[index][growth_index]+1
                gr_index=np.where(self.his_gro_count[index]> self.count_thre)[0]
                self.codebook[index]=self.codebook[index].view(-1)
                self.codebook[index][gr_index]=1
                self.codebook[index]=self.codebook[index].view(self.model_size[index][0],-1)
                print(len(gr_index),len(growth_ind),len(p_index))
                self.his_groth[index]=growth_ind
                self.his_gro_count[index][gr_index]=0

    def if_zero(self):
        cc=[]
        for index in self.convlayer:
            if index<self.convlayer[-1]:
                ww=self.feature[index].conv.weight
            if index==self.convlayer[-1]:
                ww=self.fc1.fc.weight
            if len(ww.size()) > 1:
                a = ww.data.view(self.model_length[index])
                b = a.cpu().numpy()
                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))
                cc.append(len(b) - np.count_nonzero(b))
        return cc
                

