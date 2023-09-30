from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import torch
from torch import nn as nn
from model import *
from tqdm import tqdm
import argparse
from datetime import datetime
import logging
from timm.utils import *
from spikingjelly.datasets.n_mnist import NMNIST
from timm.loss import LabelSmoothingCrossEntropy
from braincog.base.utils.criterions import *
import networkx as nx
import time
from braincog.base.learningrule.STDP import *

def randbool(size, p=0.5):
    return torch.rand(*size) < p

def calc_f2(con,device):       
    batch_size=1
    liquid_size=8000
    images=torch.load('/1000images.pt')
    labels=torch.load('/1000labels.pt')

    load_path='970.t7'


    snn = nSNN(ins=2312,
            batchsize=batch_size,
            device=device,
            liquid_size=liquid_size,
            lsm_tau=2.0,
            lsm_th=0.20,
            connectivity_matrix=randbool([liquid_size, liquid_size],p=0.01).to(device).int())

    snn.load_state_dict(torch.load(load_path,map_location={'cuda:2':device})['fc'])
    snn.con[0].load_state_dict(torch.load(load_path,map_location={'cuda:2':device})['lsm0'])

    snn.to(device)
    criterion = UnilateralMse(1.)

    optimizer = torch.optim.AdamW(snn.fc.parameters(),lr=0.001, weight_decay=1e-4)

    k=0
    sbr=0
    snn.connectivity_matrix=con
    snn.learning_rule=[]
    w2tmp=nn.Linear(liquid_size,liquid_size,bias=False,device=device)

    w2tmp.weight.data=(torch.load(load_path,map_location={'cuda:2':device})['liquid_weight'])*snn.connectivity_matrix
    snn.learning_rule.append(MutliInputSTDP(snn.node_lsm(), [snn.con[0], w2tmp])) 
    snn.eval()
    for label,data in zip(labels,images):
        running_loss = 0
        snn.zero_grad()
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        data=data.reshape(batch_size,data.shape[0],-1) 
        output = snn(data)
        # print(torch.argmax(output)==label)

        out_liquid=snn.firing_tw.squeeze(-2)

        mupost=torch.matmul(con,out_liquid.unsqueeze(-1))
        mupre=torch.matmul(con.t(),out_liquid.unsqueeze(-1))
        for t in range(snn.tw):
            if t>5 and t<snn.tw-5:
                mupost[t] = torch.sum(mupost[t+1:t+5],dim=0)
                mupre[t] = torch.sum(mupre[t-5:t-1],dim=0)
        br=mupost/mupre
        br[torch.isnan(br)] = 0
        br[torch.isinf(br)] = 0
        br=(torch.sum(out_liquid*br.squeeze(-1),dim=1)/torch.sum(out_liquid,dim=1)).sum()/snn.tw
        if torch.isnan(br):
            continue
        k+=1
        if k==500:
            break

        sbr+=br

        snn.reset()
    # print(sbr/k)

    return sbr/k

    
