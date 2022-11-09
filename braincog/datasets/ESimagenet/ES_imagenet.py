# -*- coding: utf-8 -*-            
# Time : 2022/11/1 11:06
# Author : Regulus
# FileName: ES_imagenet.py
# Explain: 
# Software: PyCharm
import numpy as np
import torch
import linecache
import torch.utils.data as data


class ESImagenet_Dataset(data.Dataset):
    def __init__(self, mode, data_set_path='/data/dvsimagenet/', transform=None):
        super().__init__()
        self.mode = mode
        self.filenames = []
        self.trainpath = data_set_path + 'train'
        self.testpath = data_set_path + 'val'
        self.traininfotxt = data_set_path + 'trainlabel.txt'
        self.testinfotxt = data_set_path + 'vallabel.txt'
        self.formats = '.npz'
        self.transform = transform
        if mode == 'train':
            self.path = self.trainpath
            trainfile = open(self.traininfotxt, 'r')
            for line in trainfile:
                filename, classnum, a, b = line.split()
                realname, sub = filename.split('.')
                self.filenames.append(realname + self.formats)
        else:
            self.path = self.testpath
            testfile = open(self.testinfotxt, 'r')
            for line in testfile:
                filename, classnum, a, b = line.split()
                realname, sub = filename.split('.')
                self.filenames.append(realname + self.formats)

    def __getitem__(self, index):
        if self.mode == 'train':
            info = linecache.getline(self.traininfotxt, index + 1)
        else:
            info = linecache.getline(self.testinfotxt, index + 1)
        filename, classnum, a, b = info.split()
        realname, sub = filename.split('.')
        filename = realname + self.formats
        filename = self.path + r'/' + filename
        classnum = int(classnum)
        a = int(a)
        b = int(b)
        datapos = np.load(filename)['pos'].astype(np.float64)
        dataneg = np.load(filename)['neg'].astype(np.float64)

        dy = (254 - b) // 2
        dx = (254 - a) // 2
        input = torch.zeros([2, 8, 256, 256])

        x = datapos[:, 0] + dx
        y = datapos[:, 1] + dy
        t = datapos[:, 2] - 1
        input[0, t, x, y] = 1

        x = dataneg[:, 0] + dx
        y = dataneg[:, 1] + dy
        t = dataneg[:, 2] - 1
        input[1, t, x, y] = 1

        reshape = input[:, :, 16:240, 16:240].permute(0, 1, 2, 3).contiguous()
        if self.transform is not None:
            reshape = self.transform(reshape)
        label = torch.tensor([classnum])
        return reshape, label

    def __len__(self):
        return len(self.filenames)