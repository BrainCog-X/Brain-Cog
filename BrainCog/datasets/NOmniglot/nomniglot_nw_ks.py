import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from braincog.datasets.NOmniglot.NOmniglot import NOmniglot


class NOmniglotNWayKShot(Dataset):
    '''
    get n-wway k-shot data as meta learning
    We set the sampling times of each epoch as "len(self.dataSet) // (self.n_way * (self.k_shot + self.k_query))"
    you can increase or decrease the number of epochs to determine the total training times
    '''

    def __init__(self, root, n_way, k_shot, k_query, train=True, frames_num=12, data_type='event',
                 transform=torchvision.transforms.Resize((28, 28))):
        self.dataSet = NOmniglot(root=root, train=train,
                                 frames_num=frames_num, data_type=data_type, transform=transform)
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <= 20
        self.length = 256
        self.data_cache = self.load_data_cache(self.dataSet.datadict, self.length)

    def load_data_cache(self, data_dict, length):
        '''
        The dataset is sampled randomly length times, and the address is saved to obtain
        '''
        data_cache = []
        for i in range(length):
            selected_cls = np.random.choice(len(data_dict), self.n_way, False)

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for j, cur_class in enumerate(selected_cls):
                selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                x_spts.append(np.array(data_dict[cur_class])[selected_img[:self.k_shot]])
                x_qrys.append(np.array(data_dict[cur_class])[selected_img[self.k_shot:]])
                y_spts.append([j for _ in range(self.k_shot)])
                y_qrys.append([j for _ in range(self.k_query)])

            shufflespt = np.random.choice(self.n_way * self.k_shot, self.n_way * self.k_shot, False)
            shuffleqry = np.random.choice(self.n_way * self.k_query, self.n_way * self.k_query, False)

            temp = [np.array(x_spts).reshape(-1)[shufflespt], np.array(y_spts).reshape(-1)[shufflespt],
                    np.array(x_qrys).reshape(-1)[shuffleqry], np.array(y_qrys).reshape(-1)[shuffleqry]]
            data_cache.append(temp)
        return data_cache

    def __getitem__(self, index):
        x_spts, y_spts, x_qrys, y_qrys = self.data_cache[index]
        x_sptst, y_sptst, x_qryst, y_qryst = [], [], [], []

        for i, j in zip(x_spts, y_spts):
            i, j = self.dataSet.readimage(i, j)
            x_sptst.append(i.unsqueeze(0))
            y_sptst.append(j)
        for i, j in zip(x_qrys, y_qrys):
            i, j = self.dataSet.readimage(i, j)
            x_qryst.append(i.unsqueeze(0))
            y_qryst.append(j)
        return torch.cat(x_sptst, dim=0), np.array(y_sptst), torch.cat(x_qryst, dim=0), np.array(y_qryst)

    def reset(self):
        self.data_cache = self.load_data_cache(self.dataSet.datadict, self.length)

    def __len__(self):
        return len(self.data_cache)


if __name__ == "__main__":
    db_train = NOmniglotNWayKShot('./data/', n_way=5, k_shot=1, k_query=15,
                                  frames_num=4, data_type='frequency', train=True)
    dataloadertrain = DataLoader(db_train, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
    for x_spt, y_spt, x_qry, y_qry in dataloadertrain:
        print(x_spt.shape)
    db_train.resampling()
