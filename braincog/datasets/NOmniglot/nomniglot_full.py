import torch
from torch.utils.data import Dataset, DataLoader
from braincog.datasets.NOmniglot.NOmniglot import NOmniglot


class NOmniglotfull(Dataset):
    '''
    solve few-shot learning as general classification problem,
    We combine the original training set with the test set and take 3/4 as the training set
    '''

    def __init__(self, root='data/', train=True, frames_num=4, data_type='event',
                 transform=None, target_transform=None, use_npz=False, crop=True, create=True):
        super().__init__()

        trainSet = NOmniglot(root=root, train=True, frames_num=frames_num, data_type=data_type,
                             transform=transform, target_transform=target_transform,
                             use_npz=use_npz, crop=crop, create=create)
        testSet = NOmniglot(root=root, train=False, frames_num=frames_num, data_type=data_type,
                            transform=transform, target_transform=lambda x: x + 964,
                            use_npz=use_npz, crop=crop, create=create)
        self.data = torch.utils.data.ConcatDataset([trainSet, testSet])
        if train:
            self.id = [j for j in range(len(self.data)) if j % 20 in [i for i in range(15)]]

        else:
            self.id = [j for j in range(len(self.data)) if j % 20 in [i for i in range(15, 20)]]

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        image, label = self.data[self.id[index]]
        return image, label


if __name__ == '__main__':
    db_train = NOmniglotfull('../../data/', train=True, frames_num=4, data_type='event')
    dataloadertrain = DataLoader(db_train, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
    for x_spt, y_spt, x_qry, y_qry in dataloadertrain:
        print(x_spt.shape)
