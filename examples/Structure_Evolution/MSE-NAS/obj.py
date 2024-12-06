import sys
import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from timm.models import create_model
from cell123model import NetworkCIFAR
from braincog.base.node.node import *
from braincog.utils import *
from braincog.base.utils.criterions import *
from braincog.datasets.datasets import *
from braincog.model_zoo.resnet import *
from braincog.model_zoo.convnet import *
from braincog.model_zoo.reactnet import *
from braincog.model_zoo.convxnet import *
from scipy.stats import kendalltau
from misc import utils
import micro_encoding
from misc.flops_counter import add_flops_counting_methods
from utils import data_transforms
from datetime import datetime
bits=20

def logdet(K):
    s, ld = torch.linalg.slogdet(K)
    return ld


def LSP(args,genome,train_data):

    with torch.no_grad():
        test_motifs,ids = micro_encoding.decode_motif(layers=args.layers,bits=bits,genome=genome)
        pmodel = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            dataset=args.dataset,
            step=args.step,
            encode_type=args.encode,
            node_type=eval(args.node_type),
            threshold=args.threshold,
            tau=args.tau,
            sigmoid_thres=args.sigmoid_thres,
            requires_thres_grad=args.requires_thres_grad,
            spike_output=not args.no_spike_output,
            C=args.init_channels,
            layers=args.layers,
            auxiliary=args.auxiliary,
            motif=test_motifs,
            parse_method=args.parse_method,
            act_fun=args.act_fun,
            temporal_flatten=args.temporal_flatten,
            layer_by_layer=args.layer_by_layer,
            n_groups=args.n_groups,
            cell_type=genome[-1]
        )
        pmodel.to(args.device)

        pmodel.K = torch.zeros(args.batch_size, args.batch_size,device=args.device)
        pmodel.J = torch.zeros(args.batch_size, args.batch_size,device=args.device)

        # pmodel.Cou = torch.zeros(args.batch_size, args.batch_size,device=args.device)
        pmodel.Ccosine = torch.zeros(args.batch_size, args.batch_size,device=args.device)
        pmodel.Cm = torch.zeros(args.batch_size, args.batch_size,device=args.device)
        pmodel.Cpe = torch.zeros(args.batch_size, args.batch_size,device=args.device)

        # pmodel.Cou = torch.zeros(args.batch_size,device=args.device)
        # pmodel.Ccosine = torch.zeros(args.batch_size, device=args.device)
        # pmodel.Cm = torch.zeros(args.batch_size,device=args.device)
        pmodel.num_actfun_C = 0    
        pmodel.num_actfun_K = 0    

        def computing_LSP(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            # 
            out = out.view(out.size(0), -1)
            batch_num , neuron_num = out.size()
            x = (out > 0).float()
            full_matrix = torch.ones((args.batch_size, args.batch_size)).cuda() * neuron_num
            sparsity = (x.sum(1)/neuron_num).unsqueeze(1)
            norm_K = ((sparsity @ (1-sparsity.t())) + ((1-sparsity) @ sparsity.t())) * neuron_num
            rescale_factor = torch.div(0.5* torch.ones((args.batch_size, args.batch_size)).cuda(), norm_K+1e-3)
            K1_0 = (x @ (1 - x.t()))
            K0_1 = ((1-x) @ x.t())
            K0_0 = (1-x) @ (1-x).t()
            K1_1 = (1-x) @ (1-x).t()

            K_total = (full_matrix - rescale_factor * (K0_1 + K1_0))
            J_total = (K1_1+K0_0)/(K0_1+K1_0+K1_1)
            pmodel.K = pmodel.K + K_total
            pmodel.J = pmodel.J + J_total
            pmodel.num_actfun_K += 1
            # x = x / torch.norm(x, dim=-1, keepdim=True)
            # similarity = torch.mm(x, x.T)  



            # dis_ou=torch.zeros_like(pmodel.Cou)
            dis_man=torch.zeros_like(pmodel.Cm)
            dis_cosine=torch.zeros_like(pmodel.Ccosine)


            ou_dist = nn.PairwiseDistance(p=2)
            m_dist = nn.PairwiseDistance(p=1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            # for i in range(args.batch_size):
            #     for j in range(i,args.batch_size):
            #         input1 = x[i]
            #         input2 = x[j]
            #         dis_ou[i][j] = ou_dist(input1,input2)
            #         dis_man[i][j] = m_dist(input1,input2)
            #         dis_cosine[i][j] = cos(input1,input2)
            
            # pmodel.Cou = pmodel.Cou + dis_ou
            for i in range(args.batch_size):
                temp = x[i].repeat(args.batch_size,1)
                dis_cosine[i] = cos(x,temp)
                dis_man[i] = m_dist(x,temp)
                 

                
            # pmodel.Cou = pmodel.Cou + ou_dist(x,x.flip(dims=[0]))
            # pmodel.Cm = pmodel.Cou + m_dist(x,x.flip(dims=[0]))
            pmodel.Ccosine = pmodel.Ccosine + dis_cosine
            pmodel.Cm = pmodel.Cm + dis_man
            pmodel.Cpe = pmodel.Cpe + torch.corrcoef(x / torch.norm(x, dim=-1, keepdim=True))

            pmodel.num_actfun_C += 1
            pmodel.num_actfun_K += 1



        s_ou = []
        s_m = []
        s_pe = []
        s_cos = []
        s_k = []
        s_jac=[]
        s_sum_j=[]
        repeat=2
        for name,module in pmodel.named_modules():
            if args.node_type in str(type(module)):    
                handle = module.register_forward_hook(computing_LSP)

        for j in range(repeat):
            pmodel.K = torch.zeros(args.batch_size, args.batch_size,device=args.device)
            pmodel.J = torch.zeros(args.batch_size, args.batch_size,device=args.device)

            pmodel.Ccosine = torch.zeros(args.batch_size, args.batch_size,device=args.device)
            pmodel.Cm = torch.zeros(args.batch_size, args.batch_size,device=args.device)
            pmodel.Cpe = torch.zeros(args.batch_size, args.batch_size,device=args.device)
            pmodel.num_actfun_C = 0    
            pmodel.num_actfun_K = 0    

            data_iterator = iter(train_data)
            inputs, targets = next(data_iterator)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = pmodel(inputs)
            tc=pmodel.Ccosine/pmodel.num_actfun_C
            tp=pmodel.Cpe/pmodel.num_actfun_C
            tm=pmodel.Cm/pmodel.num_actfun_C
            tj=pmodel.J/ (pmodel.num_actfun_K)
            Ccos = torch.where(torch.isnan(tc), torch.full_like(tc, 0), tc)
            Cpe = torch.where(torch.isnan(tp), torch.full_like(tp, 0), tp)
            Cm = torch.where(torch.isnan(tm), torch.full_like(tm, 0), tm)

            s_k.append(float(logdet(pmodel.K/ (pmodel.num_actfun_K))))
            s_jac.append(float(logdet(tj)))
            s_sum_j.append(float(tj.sum()))
            s_m.append(float(Cm.sum()))
            s_cos.append(float(Ccos.sum()))
            s_pe.append(float(Cpe.sum()))
    return np.mean(np.array(s_sum_j)),np.mean(np.array(s_jac)), np.mean(np.array(s_m)),np.mean(np.array(s_cos)),np.mean(np.array(s_pe)),np.mean(np.array(s_k))



