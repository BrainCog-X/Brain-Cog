import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.random import choice as npc
import random
import torch.nn.functional as F
from braincog.datasets.NOmniglot import NOmniglot


class NOmniglotTrainSet(Dataset):
    '''
    Dataloader for Siamese Net
    The pairs of similar samples are labeled as 1, and those of different samples are labeled as 0
    '''

    def __init__(self, root='data/', use_frame=True, frames_num=10, data_type='event', use_npz=False, resize=None):
        super(NOmniglotTrainSet, self).__init__()
        self.resize = resize
        self.data_type = data_type
        self.use_frame = use_frame
        self.dataSet = NOmniglot(root=root, train=True, frames_num=frames_num, data_type=data_type, use_npz=use_npz)
        self.datas, self.num_classes = self.dataSet.datadict, self.dataSet.num_classes

        np.random.seed(0)

    def __len__(self):
        '''
        Sampling upper limit, you can set the maximum sampling times when using to terminate
        '''
        return 21000000

    def __getitem__(self, index):
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx1 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx1])
        # get image from different class
        else:
            label = 0.0
            idx1 = random.randint(0, self.num_classes - 1)
            idx2 = random.randint(0, self.num_classes - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes - 1)
            image1 = random.choice(self.datas[idx1])
            image2 = random.choice(self.datas[idx2])

        if self.use_frame:
            if self.data_type == 'event':
                image1 = torch.tensor(np.load(image1)['arr_0']).float()
                image2 = torch.tensor(np.load(image2)['arr_0']).float()
            elif self.data_type == 'frequency':
                image1 = torch.tensor(np.load(image1)['arr_0']).float()
                image2 = torch.tensor(np.load(image2)['arr_0']).float()
            else:
                raise NotImplementedError

        if self.resize is not None:
            image1 = image1[:, :, 4:254, 54:304]
            image1 = F.interpolate(image1, size=(self.resize, self.resize))
            image2 = image2[:, :, 4:254, 54:304]
            image2 = F.interpolate(image2, size=(self.resize, self.resize))

        return image1, image2, torch.from_numpy(np.array([label], dtype=np.float32))


class NOmniglotTestSet(Dataset):
    '''
        Dataloader for Siamese Net

        '''

    def __init__(self, root='data/', time=1000, way=20, shot=1, query=1, use_frame=True, frames_num=10, data_type='event', use_npz=True, resize=None):
        super(NOmniglotTestSet, self).__init__()
        self.resize = resize
        self.use_frame = use_frame
        self.time = time         # Sampling times
        self.way = way
        self.shot = shot
        self.query = query
        self.img1 = None         # Fix test sample while sampling support set
        self.c1 = None           # Fixed categories when sampling multiple samples
        self.c2 = None
        self.select_class = []   # selected classes
        self.select_sample = []  # selected samples

        self.data_type = data_type
        np.random.seed(0)
        self.dataSet = NOmniglot(root=root, train=False, frames_num=frames_num, data_type=data_type, use_npz=use_npz)
        self.datas, self.num_classes = self.dataSet.datadict, self.dataSet.num_classes

    def __len__(self):
        '''
        In general, the total number of test tasks is 1000.
        Since one test sample is collected at a time, way * shot support samples are used for each test
        '''
        return self.time * self.way * self.shot

    def __getitem__(self, index):
        '''
        The 0th sample of each way*shot is used for query and recorded in the selected sample
        to achieve the effect of selecting K +1
        '''
        idx = index % (self.way * self.shot)
        # generate image pair from same class
        if idx == 0:  #
            self.select_class = []
            self.c1 = random.randint(0, self.num_classes - 1)
            self.c2 = self.c1
            sind = random.randint(0, len(self.datas[self.c1]) - 1)
            self.select_sample.append(sind)
            self.img1 = self.datas[self.c1][sind]

            sind = random.randint(0, len(self.datas[self.c2]) - 1)
            while sind in self.select_sample:
                sind = random.randint(0, len(self.datas[self.c2]) - 1)
            img2 = self.datas[self.c1][sind]
            self.select_sample.append(sind)
            self.select_class.append(self.c1)
        # generate image pair from different class
        else:
            if index % self.shot == 0:
                self.c2 = random.randint(0, self.num_classes - 1)
                while self.c2 in self.select_class:  # self.c1 == c2:
                    self.c2 = random.randint(0, self.num_classes - 1)
                self.select_class.append(self.c2)
                self.select_sample = []
            sind = random.randint(0, len(self.datas[self.c2]) - 1)
            while sind in self.select_sample:
                sind = random.randint(0, len(self.datas[self.c2]) - 1)
            img2 = self.datas[self.c2][sind]
            self.select_sample.append(sind)

        if self.use_frame:
            if self.data_type == 'event':
                img1 = torch.tensor(np.load(self.img1)['arr_0']).float()
                img2 = torch.tensor(np.load(img2)['arr_0']).float()
            elif self.data_type == 'frequency':
                img1 = torch.tensor(np.load(self.img1)['arr_0']).float()
                img2 = torch.tensor(np.load(img2)['arr_0']).float()
            else:
                raise NotImplementedError

        if self.resize is not None:
            img1 = img1[:, :, 4:254, 54:304]
            img1 = F.interpolate(img1, size=(self.resize, self.resize))
            img2 = img2[:, :, 4:254, 54:304]
            img2 = F.interpolate(img2, size=(self.resize, self.resize))
        return img1, img2


if __name__ == '__main__':
    data_type = 'frequency'
    T = 4
    trainSet = NOmniglotTrainSet(root='data/', use_frame=True, frames_num=T, data_type=data_type, use_npz=True, resize=105)
    testSet = NOmniglotTestSet(root='data/', time=1000, way=5, shot=1, use_frame=True, frames_num=T,
                               data_type=data_type, use_npz=True, resize=105)
    trainLoader = DataLoader(trainSet, batch_size=48, shuffle=False, num_workers=4)
    testLoader = DataLoader(testSet, batch_size=5 * 1, shuffle=False, num_workers=4)
    for batch_id, (img1, img2) in enumerate(testLoader, 1):
        # img1.shape [batch, T, 2, H, W]
        print(batch_id)
        break

    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        # img1.shape [batch, T, 2, H, W]
        print(batch_id)
        break
