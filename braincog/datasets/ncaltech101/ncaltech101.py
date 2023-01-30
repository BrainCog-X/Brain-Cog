# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/1/30 21:28
# User      : yu
# Product   : PyCharm
# Project   : BrainCog
# File      : ncaltech101.py
# explain   :
import os
import numpy as np

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive


class NCALTECH101(Dataset):
    """N-CALTECH101 dataset <https://www.garrickorchard.com/datasets/n-caltech101>. Events have (xytp) ordering.
    ::

        @article{orchard2015converting,
          title={Converting static image datasets to spiking neuromorphic datasets using saccades},
          author={Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K and Thakor, Nitish},
          journal={Frontiers in neuroscience},
          volume={9},
          pages={437},
          year={2015},
          publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    url = "https://data.mendeley.com/public-files/datasets/cy6cvx3ryv/files/36b5c52a-b49d-4853-addb-a836a8883e49/file_downloaded"
    filename = "N-Caltech101-archive.zip"
    file_md5 = "66201824eabb0239c7ab992480b50ba3"
    data_filename = "N-Caltech101-archive.zip"
    folder_name = "Caltech101"
    cls_count = [467,
                 435, 200, 798, 55, 800, 42, 42, 47, 54, 46,
                 33, 128, 98, 43, 85, 91, 50, 43, 123, 47,
                 59, 62, 107, 47, 69, 73, 70, 50, 51, 57,
                 67, 52, 65, 68, 75, 64, 53, 64, 85, 67,
                 67, 45, 34, 34, 51, 99, 100, 42, 54, 88,
                 80, 31, 64, 86, 114, 61, 81, 78, 41, 66,
                 43, 40, 87, 32, 76, 55, 35, 39, 47, 38,
                 45, 53, 34, 57, 82, 59, 49, 40, 63, 39,
                 84, 57, 35, 64, 45, 86, 59, 64, 35, 85,
                 49, 86, 75, 239, 37, 59, 34, 56, 39, 60]
    # length = 8242
    length = 8709

    sensor_size = None  # all recordings are of different size
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None):
        super(NCALTECH101, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        classes = {
            'BACKGROUND_Google': 0,
            'Faces_easy': 1,
            'Leopards': 2,
            'Motorbikes': 3,
            'accordion': 4,
            'airplanes': 5,
            'anchor': 6,
            'ant': 7,
            'barrel': 8,
            'bass': 9,
            'beaver': 10,
            'binocular': 11,
            'bonsai': 12,
            'brain': 13,
            'brontosaurus': 14,
            'buddha': 15,
            'butterfly': 16,
            'camera': 17,
            'cannon': 18,
            'car_side': 19,
            'ceiling_fan': 20,
            'cellphone': 21,
            'chair': 22,
            'chandelier': 23,
            'cougar_body': 24,
            'cougar_face': 25,
            'crab': 26,
            'crayfish': 27,
            'crocodile': 28,
            'crocodile_head': 29,
            'cup': 30,
            'dalmatian': 31,
            'dollar_bill': 32,
            'dolphin': 33,
            'dragonfly': 34,
            'electric_guitar': 35,
            'elephant': 36,
            'emu': 37,
            'euphonium': 38,
            'ewer': 39,
            'ferry': 40,
            'flamingo': 41,
            'flamingo_head': 42,
            'garfield': 43,
            'gerenuk': 44,
            'gramophone': 45,
            'grand_piano': 46,
            'hawksbill': 47,
            'headphone': 48,
            'hedgehog': 49,
            'helicopter': 50,
            'ibis': 51,
            'inline_skate': 52,
            'joshua_tree': 53,
            'kangaroo': 54,
            'ketch': 55,
            'lamp': 56,
            'laptop': 57,
            'llama': 58,
            'lobster': 59,
            'lotus': 60,
            'mandolin': 61,
            'mayfly': 62,
            'menorah': 63,
            'metronome': 64,
            'minaret': 65,
            'nautilus': 66,
            'octopus': 67,
            'okapi': 68,
            'pagoda': 69,
            'panda': 70,
            'pigeon': 71,
            'pizza': 72,
            'platypus': 73,
            'pyramid': 74,
            'revolver': 75,
            'rhino': 76,
            'rooster': 77,
            'saxophone': 78,
            'schooner': 79,
            'scissors': 80,
            'scorpion': 81,
            'sea_horse': 82,
            'snoopy': 83,
            'soccer_ball': 84,
            'stapler': 85,
            'starfish': 86,
            'stegosaurus': 87,
            'stop_sign': 88,
            'strawberry': 89,
            'sunflower': 90,
            'tick': 91,
            'trilobite': 92,
            'umbrella': 93,
            'watch': 94,
            'water_lilly': 95,
            'wheelchair': 96,
            'wild_cat': 97,
            'windsor_chair': 98,
            'wrench': 99,
            'yin_yang': 100,
        }

        # if not self._check_exists():
            # self.download()
            # extract_archive(os.path.join(self.location_on_system, self.data_filename))

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            # if 'BACKGROUND_Google' in path:
            #     continue
            for file in files:
                if file.endswith("bin"):
                    self.data.append(path + "/" + file)
                    label_name = os.path.basename(path)

                    if isinstance(label_name, bytes):
                        label_name = label_name.decode()
                    self.targets.append(classes[label_name])

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = read_mnist_file(self.data[index], dtype=self.dtype)
        target = self.targets[index]
        events["x"] -= events["x"].min()
        events["y"] -= events["y"].min()
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present() and self._folder_contains_at_least_n_files_of_type(
            8709, ".bin"
        )
