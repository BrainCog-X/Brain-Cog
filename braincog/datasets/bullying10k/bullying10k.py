import os
import numpy as np
from numpy.lib import recfunctions
import scipy.io as scio
from typing import Tuple, Any, Optional
from tonic.dataset import Dataset
from tonic.download_utils import extract_archive
import dv


class BULLYINGDVS(Dataset):
    classes = ["fingerguess", "greeting", "hairgrabs", "handshake", "kicking",
               "punching", "pushing", "slapping", "strangling", "walking"]
    class_dict = {cls: idx for idx, cls in enumerate(classes)}

    sensor_size = (346, 260, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None):
        super(BULLYINGDVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.aedat4 = True

        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("aedat4"):
                    self.data.append(path + "/" + file)
                    self.targets.append(self.class_dict[path.split('/')[-2]])

                if file.endswith("npy"):
                    self.aedat4 = False
                    self.data.append(path + "/" + file)
                    self.targets.append(self.class_dict[path.split('/')[-2]])


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        if self.aedat4:
            events, target = dv.AedatFile(self.data[index])['events'], self.targets[index]
            events = np.concatenate([event for event in events.numpy()])
        else:
            events = np.concatenate(np.load(self.data[index], allow_pickle=True))

        events = np.column_stack(
            [
                events['timestamp'] - events['timestamp'][0],
                events['x'],
                events['y'],
                events['polarity']
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
        return True
