import os.path as osp
import numpy as np
import glob

from albumentations.pytorch import ToTensorV2

from torchvision import datasets, transforms
import torch
from inclearn.tools.cutout import Cutout
from inclearn.tools.autoaugment_extra import ImageNetPolicy


def get_datasets(dataset_names):
    return [get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def get_dataset(dataset_name):
    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif "imagenet100" in dataset_name:
        return iImageNet100
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [ToTensorV2()]
    class_order = None


class iCIFAR10(DataHandler):
    base_dataset_cls = datasets.cifar.CIFAR10
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data_folder, train, is_fine_label=False):
        self.base_dataset = self.base_dataset_cls(data_folder, train=train, download=True)
        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.n_cls = 10

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        return [4, 0, 2, 5, 8, 3, 1, 6, 9, 7]


class iCIFAR100(iCIFAR10):
    label_list = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
        'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
        'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
        'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
        'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ]
    base_dataset_cls = datasets.cifar.CIFAR100
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    def __init__(self, data_folder, train, is_fine_label=False):
        self.base_dataset = self.base_dataset_cls(data_folder, train=train, download=True)
        self.data = self.base_dataset.data
        self.targets = self.base_dataset.targets
        self.n_cls = 100
        self.transform_type = 'torchvision'

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
        return [
                87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45,
                88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6,
                46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
                40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39
            ]


class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [ToTensorV2()]
    class_order = None


class iImageNet100(DataHandler):

    base_dataset_cls = datasets.ImageFolder
    transform_type = 'torchvision'
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])
    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_folder, train, is_fine_label=False):
        if train is True:
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "train"))
        else:
            self.base_dataset = self.base_dataset_cls(osp.join(data_folder, "val"))

        self.data, self.targets = zip(*self.base_dataset.samples)
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.n_cls = 200

    @property
    def is_proc_inc_data(self):
        return False

    @classmethod
    def class_order(cls, trial_i):
       return [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33,
                        168, 156, 178, 108, 123, 184, 190, 165, 174, 176, 140, 189, 103,
       192, 155, 109, 126, 180, 143, 138, 158, 170, 177, 101, 185, 119,
       117, 150, 128, 153, 113, 181, 145, 182, 106, 159, 183, 116, 115,
       144, 191, 141, 172, 160, 179, 152, 120, 110, 131, 154, 137, 195,
       114, 171, 196, 198, 197, 102, 164, 166, 142, 122, 135, 186, 124,
       134, 187, 121, 199, 100, 188, 127, 118, 194, 111, 112, 147, 125,
       130, 146, 162, 169, 136, 161, 107, 163, 175, 105, 132, 104, 151,
       148, 173, 193, 139, 167, 129, 149, 157, 133,

       268, 256, 278, 208, 223, 284, 290, 265, 274, 276, 240, 289, 203,
       292, 255, 209, 226, 280, 243, 238, 258, 270, 277, 201, 285, 219,
       217, 250, 228, 253, 213, 281, 245, 282, 206, 259, 283, 216, 215,
       244, 291, 241, 272, 260, 279, 252, 220, 210, 231, 254, 237, 295,
       214, 271, 296, 298, 297, 202, 264, 266, 242, 222, 235, 286, 224,
       234, 287, 221, 299, 200, 288, 227, 218, 294, 211, 212, 247, 225,
       230, 246, 262, 269, 236, 261, 207, 263, 275, 205, 232, 204, 251,
       248, 273, 293, 239, 267, 229, 249, 257, 233, 368, 356, 378, 308,
       323, 384, 390, 365, 374, 376, 340, 389, 303, 392, 355, 309, 326,
       380, 343, 338, 358, 370, 377, 301, 385, 319, 317, 350, 328, 353,
       313, 381, 345, 382, 306, 359, 383, 316, 315, 344, 391, 341, 372,
       360, 379, 352, 320, 310, 331, 354, 337, 395, 314, 371, 396, 398,
       397, 302, 364, 366, 342, 322, 335, 386, 324, 334, 387, 321, 399,
       300, 388, 327, 318, 394, 311, 312, 347, 325, 330, 346, 362, 369,
       336, 361, 307, 363, 375, 305, 332, 304, 351, 348, 373, 393, 339,
       367, 329, 349, 357, 333,

       468, 456, 478, 408, 423, 484, 490, 465, 474, 476, 440, 489, 403,
       492, 455, 409, 426, 480, 443, 438, 458, 470, 477, 401, 485, 419,
       417, 450, 428, 453, 413, 481, 445, 482, 406, 459, 483, 416, 415,
       444, 491, 441, 472, 460, 479, 452, 420, 410, 431, 454, 437, 495,
       414, 471, 496, 498, 497, 402, 464, 466, 442, 422, 435, 486, 424,
       434, 487, 421, 499, 400, 488, 427, 418, 494, 411, 412, 447, 425,
       430, 446, 462, 469, 436, 461, 407, 463, 475, 405, 432, 404, 451,
       448, 473, 493, 439, 467, 429, 449, 457, 433, 568, 556, 578, 508,
       523, 584, 590, 565, 574, 576, 540, 589, 503, 592, 555, 509, 526,
       580, 543, 538, 558, 570, 577, 501, 585, 519, 517, 550, 528, 553,
       513, 581, 545, 582, 506, 559, 583, 516, 515, 544, 591, 541, 572,
       560, 579, 552, 520, 510, 531, 554, 537, 595, 514, 571, 596, 598,
       597, 502, 564, 566, 542, 522, 535, 586, 524, 534, 587, 521, 599,
       500, 588, 527, 518, 594, 511, 512, 547, 525, 530, 546, 562, 569,
       536, 561, 507, 563, 575, 505, 532, 504, 551, 548, 573, 593, 539,
       567, 529, 549, 557, 533,

       668, 656, 678, 608, 623, 684, 690, 665, 674, 676, 640, 689, 603,
       692, 655, 609, 626, 680, 643, 638, 658, 670, 677, 601, 685, 619,
       617, 650, 628, 653, 613, 681, 645, 682, 606, 659, 683, 616, 615,
       644, 691, 641, 672, 660, 679, 652, 620, 610, 631, 654, 637, 695,
       614, 671, 696, 698, 697, 602, 664, 666, 642, 622, 635, 686, 624,
       634, 687, 621, 699, 600, 688, 627, 618, 694, 611, 612, 647, 625,
       630, 646, 662, 669, 636, 661, 607, 663, 675, 605, 632, 604, 651,
       648, 673, 693, 639, 667, 629, 649, 657, 633, 768, 756, 778, 708,
       723, 784, 790, 765, 774, 776, 740, 789, 703, 792, 755, 709, 726,
       780, 743, 738, 758, 770, 777, 701, 785, 719, 717, 750, 728, 753,
       713, 781, 745, 782, 706, 759, 783, 716, 715, 744, 791, 741, 772,
       760, 779, 752, 720, 710, 731, 754, 737, 795, 714, 771, 796, 798,
       797, 702, 764, 766, 742, 722, 735, 786, 724, 734, 787, 721, 799,
       700, 788, 727, 718, 794, 711, 712, 747, 725, 730, 746, 762, 769,
       736, 761, 707, 763, 775, 705, 732, 704, 751, 748, 773, 793, 739,
       767, 729, 749, 757, 733,

       868, 856, 878, 808, 823, 884, 890, 865, 874, 876, 840, 889, 803,
       892, 855, 809, 826, 880, 843, 838, 858, 870, 877, 801, 885, 819,
       817, 850, 828, 853, 813, 881, 845, 882, 806, 859, 883, 816, 815,
       844, 891, 841, 872, 860, 879, 852, 820, 810, 831, 854, 837, 895,
       814, 871, 896, 898, 897, 802, 864, 866, 842, 822, 835, 886, 824,
       834, 887, 821, 899, 800, 888, 827, 818, 894, 811, 812, 847, 825,
       830, 846, 862, 869, 836, 861, 807, 863, 875, 805, 832, 804, 851,
       848, 873, 893, 839, 867, 829, 849, 857, 833, 968, 956, 978, 908,
       923, 984, 990, 965, 974, 976, 940, 989, 903, 992, 955, 909, 926,
       980, 943, 938, 958, 970, 977, 901, 985, 919, 917, 950, 928, 953,
       913, 981, 945, 982, 906, 959, 983, 916, 915, 944, 991, 941, 972,
       960, 979, 952, 920, 910, 931, 954, 937, 995, 914, 971, 996, 998,
       997, 902, 964, 966, 942, 922, 935, 986, 924, 934, 987, 921, 999,
       900, 988, 927, 918, 994, 911, 912, 947, 925, 930, 946, 962, 969,
       936, 961, 907, 963, 975, 905, 932, 904, 951, 948, 973, 993, 939,
       967, 929, 949, 957, 933

        ]

