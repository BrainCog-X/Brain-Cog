import random
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# from torch.autograd import  Variable


class StateDataset:
    # initial
    def __init__(self, mode, num):
        self.state = np.loadtxt(mode, dtype=np.int)
        self.num = 1
        self.state = self.state.reshape(num, 5*self.num , -1)

    #data:A label:B
    def __getitem__(self, item):
        state = self.state[item]

        state_A = state[:,0:5*self.num]
        state_A = np.expand_dims(state_A, axis=0)
        state_B = state[:, 5*self.num:10*self.num]
        state_B = np.expand_dims(state_B, axis=0)

        return {"A":state_A, "B":state_B}

    #the number of data
    def __len__(self):
        return len(self.state)

# def get_dataloader(self):

def get_dataloader(mode, num, batch):
    train_dataset = StateDataset(mode, num)
    train_loader = DataLoader(train_dataset, batch, shuffle=True)
    return train_loader
