from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
import numpy as np
import torch
from torch import nn as nn
from mnistmodel import *
from tqdm import tqdm
import argparse
from datetime import datetime
import logging
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy
from braincog.base.utils import UnilateralMse, MixLoss
from braincog.base.learningrule.STDP import *

device='cuda:7'

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer   


batch_size=100
liquid_size=8000

learning_rate = 1e-3
num_epochs = 100  # max epoch

data_path = '/data'  
load_path=''
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

snn = SNN(ins=784,
        batchsize=batch_size,
        device=device,
        liquid_size=liquid_size,
        lsm_tau=lsm_tau,
        lsm_th=lsm_th)
snn.load_state_dict(torch.load(load_path)['fc'])
snn.learning_rule=[]
snn.con[0].load_state_dict(torch.load(load_path)['lsm0'])
w2tmp=nn.Linear(liquid_size,liquid_size,bias=False,device=device)
snn.connectivity_matrix=torch.load(load_path)['connectivity_matrix'].to(device)
w2tmp.weight.data=(torch.load(load_path)['liquid_weight'].to(device))*snn.connectivity_matrix
snn.learning_rule.append(MutliInputSTDP(snn.node_lsm(), [snn.con[0], w2tmp]))  # pm
snn.eval()
snn.to(device)

class LabelSmoothingBCEWithLogitsLoss(nn.Module):

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.BCELoss = nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        target = torch.eye(x.shape[-1], device=x.device)[target]
        nll = torch.ones_like(x) / x.shape[-1]
        return self.BCELoss(x, target) * self.confidence + self.BCELoss(x, nll) * self.smoothing


ls = 'mse'

if ls == 'ce':
    criterion = nn.CrossEntropyLoss()
elif ls == 'bce':
    criterion = nn.BCEWithLogitsLoss()
elif ls == 'mse':
    criterion = UnilateralMse(1.)
elif ls == 'sce':
    criterion = LabelSmoothingCrossEntropy()
elif ls == 'sbce':
    criterion = LabelSmoothingBCEWithLogitsLoss()
elif ls == 'umse':
    criterion = UnilateralMse(.5)

optimizer = torch.optim.AdamW(snn.fc.parameters(),lr=0.001, weight_decay=1e-4)

l=[]
best_acc=0
for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        outputs = snn(images)
        labels=labels.to(device)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()

        optimizer.step()
        snn.reset()
        if (i + 1) % 100 == 0:
            running_loss = 0

    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.float().to(device)
        snn.zero_grad()
        optimizer.zero_grad()
        outputs = snn(inputs)
        targets=targets.to(device)
        loss = criterion(outputs, targets)

        _, predicted = outputs.max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        snn.reset()
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    print('Test Accuracy: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    if best_acc < acc:
        best_acc = acc
    print(best_acc)
    l.append(best_acc)



