import torch
from torchvision import datasets, transforms
import time
import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import time
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from PIL import Image
import numpy as np
 

from torchvision import utils
import os


seed = 1000

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def save_image(image, filename):  
    assert len(image.shape) == 3, "The image must have only three dimensions of C,W,H."
    utils.save_image(image, filename)
    



def get_mnist_data(train = False, batch_size = 100):
    path = './datasets/' # might need to change based on where to call this function
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  
    transform = transforms.Compose([transforms.ToTensor()])
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=False, transform=transform),
                batch_size=batch_size, shuffle=False)
        return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, download=False, transform=transform),
                batch_size=batch_size, shuffle=False)
        return test_loader





def get_silhouette_data(path):
    '''
    path: dir path to the silhouette image samples of 16-clas-ImageNet
    '''
    labels = os.listdir(path)
    
    datasets = []
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = transforms.Compose([transforms.ToTensor()])
    for label in labels:
        for img_name in os.listdir(f"{path}/{label}"):
            img_path = f"{path}/{label}/{img_name}"
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0)

            datasets.append((img, label))
    return datasets


def ag_distort_28(imgs, threshold=0, interval=4, phase=2, direction=(1,0)):
    #return imgs
    assert len(imgs.shape) == 4, "The images must have four dimensions of B,C,W,H."
    B,C,W,H = imgs.shape
    mask_fg = (imgs>threshold).float()  
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)
  
    for w in range(W):
        for h in range(H):
            if (direction[0]*w+direction[1]*h)%interval==0:
                gratings_fg[:,:,w,h] = 1
            if (direction[0]*w+direction[1]*h)%interval==phase:
                gratings_bg[:,:,w,h] = 1
    masked_gratings_fg = mask_fg*gratings_fg
    masked_gratings_bg = mask_bg*gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg
    return ag_image

def transform_224(imgs):
    imgs = torch.nn.functional.interpolate(imgs, scale_factor = 8, mode = 'bilinear', align_corners = False)
    imgs = torch.cat([imgs, imgs, imgs], dim=1)
    return imgs   

def ag_distort_224(imgs, threshold=0, interval=8, phase=4, direction=(1,0)):
    assert len(imgs.shape) == 4, "The images must have four dimensions of C,W,H."   
    imgs = torch.nn.functional.interpolate(imgs, scale_factor = 8, mode = 'bilinear', align_corners = False)
    imgs = torch.cat([imgs, imgs, imgs], dim=1)
    #return imgs
    B,C,W,H = imgs.shape
    mask_fg = (imgs>threshold).float()  
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)
  
    for w in range(W):
        for h in range(H):
            if (direction[0]*w+direction[1]*h)%interval==0:
                gratings_fg[:,:,w,h] = 1
            if (direction[0]*w+direction[1]*h)%interval==phase:
                gratings_bg[:,:,w,h] = 1
    masked_gratings_fg = mask_fg*gratings_fg
    masked_gratings_bg = mask_bg*gratings_bg
    ag_image = masked_gratings_fg + masked_gratings_bg
    return ag_image


def ag_distort_silhouette(imgs, threshold=0.5, interval=2, phase=1, direction=(1,0)):

    assert len(imgs.shape) == 4, "The image must have only three dimensions of C,W,H."
    #imgs = torch.nn.functional.interpolate(imgs, scale_factor = 2, mode = 'bilinear', align_corners = False)
    B,C,W,H = imgs.shape
    mask_fg = (imgs<threshold).float()
    mask_bg = 1 - mask_fg
    gratings_fg = torch.zeros_like(imgs)
    gratings_bg = torch.zeros_like(imgs)
    for w in range(W):
        for h in range(H):
            if (direction[0]*w+direction[1]*h)%interval==0:
                gratings_fg[:,:,w,h] = 1
            if (direction[0]*w+direction[1]*h)%interval==phase:
                gratings_bg[:,:,w,h] = 1
    ag_images = mask_fg*gratings_fg + mask_bg*gratings_bg
    #transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #transform = transforms.Compose([]) 
    #ag_images[0] = transform(ag_images[0])
    return ag_images