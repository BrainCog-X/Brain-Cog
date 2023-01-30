# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2023/1/30 20:54
# User      : yu
# Product   : PyCharm
# Project   : BrainCog
# File      : hmdb_dvs.py
# explain   :

import os
import numpy as np
from numpy.lib import recfunctions
import scipy.io as scio
from typing import Tuple, Any, Optional
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive

class HMDBDVS(Dataset):
    """ASL-DVS dataset <https://github.com/PIX2NVS/NVS2Graph>. Events have (txyp) ordering.
    ::

        @inproceedings{bi2019graph,
            title={Graph-based Object Classification for Neuromorphic Vision Sensing},
            author={Bi, Y and Chadha, A and Abbas, A and and Bourtsoulatze, E and Andreopoulos, Y},
            booktitle={2019 IEEE International Conference on Computer Vision (ICCV)},
            year={2019},
            organization={IEEE}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    sensor_size = (240, 180, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None):
        super(HMDBDVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        if not self._check_exists():
            raise NotImplementedError(
                'Please manually download the dataset from'
                ' https://www.dropbox.com/sh/ie75dn246cacf6n/AACoU-_zkGOAwj51lSCM0JhGa?dl=0 '
                'and extract it to {}'.format(self.location_on_system))

        classes = os.listdir(self.location_on_system)
        self.int_classes = dict(zip(classes, range(len(classes))))

        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("mat"):
                    fsize = os.path.getsize(path + '/' + file) / float(1024)
                    if fsize < 1:
                        # print('{} size {} K'.format(file, fsize))
                        continue
                    self.data.append(path + "/" + file)
                    self.targets.append(self.int_classes[path.split('/')[-1]])

        self.length = self.__len__()
        self.cls_count = np.bincount(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        events, target = scio.loadmat(self.data[index]), self.targets[index]
        events = np.column_stack(
            [
                events["ts"],
                events["x"],
                self.sensor_size[1] - 1 - events["y"],
                events["pol"],
            ]
        )
        events = np.lib.recfunctions.unstructured_to_structured(events, self.dtype)
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._folder_contains_at_least_n_files_of_type(
            6765, ".mat"
        )
