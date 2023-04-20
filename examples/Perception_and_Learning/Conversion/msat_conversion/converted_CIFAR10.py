# -*- coding: utf-8 -*-            
# Time : 2023/4/19 15:56
# Author : Regulus
# FileName: converted_CIFAR10.py
# Explain: 
# Software: PyCharm

import sys
sys.path.append('..')
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import matplotlib
# matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from braincog.utils import setup_seed
import os
from examples.Perception_and_Learning.Conversion.msat_conversion.CIFAR10_VGG16 import VGG16
import argparse
from braincog.datasets.datasets import get_cifar10_data
from braincog.base.conversion import Convertor, FolderPath


parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=256, type=int, help='simulation time')
parser.add_argument('--p', default=1, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=1, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--lateral_inhi', default=True, type=bool, help='LIPooling')
parser.add_argument('--data_norm', default=True, type=bool, help=' whether use data norm or not')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--device', default='7', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--train_batch', default=512, type=int, help='batch size for get max')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=23, type=int, help='seed')
parser.add_argument('--useDET', action='store_true', default=False, help='use DET')
parser.add_argument('--useDTT', action='store_true', default=False, help='use DTT')
parser.add_argument('--useSC', action='store_true', default=False, help='use SpikeConfidence')
args = parser.parse_args()

def evaluate_snn(test_iter, snn, device=None, duration=50):
    folder_path = "./result_conversion_{}/snn_timestep{}_p{}_LIPooling{}_Burst{}".format(
            args.model_name, duration, args.p, args.lateral_inhi, args.gamma)
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)
    snn.eval()
    FolderPath.folder_path = folder_path
    accs = []
    for ind, (test_x, test_y) in enumerate(tqdm(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        with torch.no_grad():
            snn.reset()
            acc = []
            for t in range(duration):
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)
        # break
        accs.append(np.array(acc))

    if True:
        f = open('{}/result.txt'.format(folder_path), 'w')
        f.write("Setting Arguments.. : {}\n".format(args))
        accs = np.array(accs).mean(axis=0)
        for iii in range(256):
            if iii == 0 or iii == 3 or iii == 7 or (iii + 1) % 16 == 0:
                f.write("timestep {}:{}\n".format(str(iii+1).zfill(3), accs[iii]))
        f.write("max accs: {}, timestep:{}\n".format(max(accs), np.where(accs == max(accs))))
        f.close()
        accs = torch.from_numpy(accs)
        torch.save(accs, "{}/accs.pth".format(folder_path))

if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    setup_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'

    train_iter, _, _, _ = get_cifar10_data(args.train_batch, same_da=True)
    _, test_iter, _, _ = get_cifar10_data(args.batch_size, same_da=True)

    if args.model_name == 'vgg16':
        net = VGG16()
        # net.load_state_dict(torch.load("./CIFAR10_VGG16.pth", map_location=device))

    net.eval()
    net = net.to(device)

    converter = Convertor(dataloader=train_iter,
                          device=device,
                          p=1.0,
                          channelnorm=False,
                          lipool=True,
                          gamma=1,
                          soft_mode=True,
                          merge=True,
                          batch_num=1,
                          spicalib=False,
                          useDET=args.useDET,
                          useDTT=args.useDTT,
                          useSC=args.useSC
                          )
    snn = converter(net)

    evaluate_snn(test_iter, snn, device, duration=args.T)
