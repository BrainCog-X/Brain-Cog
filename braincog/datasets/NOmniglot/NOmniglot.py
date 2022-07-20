from torch.utils.data import Dataset
from braincog.datasets.NOmniglot.utils import *


class NOmniglot(Dataset):
    def __init__(self, root='data/', frames_num=12, train=True, data_type='event',
                 transform=None, target_transform=None, use_npz=False, crop=True, create=True, thread_num=16):
        super().__init__()
        self.crop = crop
        self.data_type = data_type
        self.use_npz = use_npz
        self.transform = transform
        self.target_transform = target_transform
        events_npy_root = os.path.join(root, 'events_npy', 'background' if train else "evaluation")

        frames_root = os.path.join(root, f'fnum_{frames_num}_dtype_{data_type}_npz_{use_npz}',
                                   'background' if train else "evaluation")

        if not os.path.exists(frames_root) and create:
            if not os.path.exists(events_npy_root) and create:
                os.makedirs(events_npy_root)
                print('creating event data..')
                convert_aedat4_dir_to_events_dir(root, train)
            else:
                print(f'npy format events data root {events_npy_root}, already exists')

            os.makedirs(frames_root)
            print('creating frames data..')
            convert_events_dir_to_frames_dir(events_npy_root, frames_root, '.npy', frames_num, data_type,
                                             thread_num=thread_num, compress=use_npz)
        else:
            print(f'frames data root {frames_root} already exists.')

        self.datadict, self.num_classes = list_class_files(events_npy_root, frames_root, True, use_npz=use_npz)

        self.datalist = []
        for i in self.datadict:
            self.datalist.extend([(j, i) for j in self.datadict[i]])

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        image, label = self.datalist[index]
        image, label = self.readimage(image, label)
        return image, label

    def readimage(self, image, label):
        if self.use_npz:
            image = torch.tensor(np.load(image)['arr_0']).float()
        else:
            image = torch.tensor(np.load(image)).float()
        if self.crop:
            image = image[:, :, 4:254, 54:304]
        if self.transform is not None: image = self.transform(image)
        if self.target_transform is not None: label = self.target_transform(label)
        return image, label



