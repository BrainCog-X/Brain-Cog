# -*- coding: utf-8 -*-            
# Time : 2023/2/14 11:52
# Author : Regulus
# FileName: main_visual_losslandscape.py
# Explain:
# Software: PyCharm
import sys

import tqdm

from loss_landscape.plot_surface import *

from Pytorch_Grad_Cam.cam import *

import argparse
import math
import time
import CKA
import numpy
import timm.models
import random as rd
import yaml
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from braincog.base.node.node import *
from braincog.utils import *
from braincog.base.utils.criterions import *
from braincog.datasets.datasets import *
from braincog.model_zoo.resnet import *
from braincog.model_zoo.convnet import *
from braincog.model_zoo.vgg_snn import VGG_SNN
from braincog.model_zoo.resnet19_snn import resnet19
from braincog.utils import save_feature_map, setup_seed
from braincog.base.utils.visualization import plot_tsne_3d, plot_tsne, plot_confusion_matrix
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from rgb_hsv import RGB_HSV
import matplotlib.pyplot as plt
from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from copy import deepcopy

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='SNN Training and Evaluating')

# Model parameters
parser.add_argument('--source-dataset', default='cifar10', type=str)
parser.add_argument('--target-dataset', default='dvsc10', type=str)
parser.add_argument('--model', default='cifar_convnet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--eval_checkpoint', default='', type=str, metavar='PATH',
                    help='path to eval checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

# Dataset parameters for static datasets
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='inputs image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')

# Dataloader parameters
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='inputs batch size for training (default: 128)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.01,
                    help='weight decay (default: 0.01 for adamw)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--adam-epoch', type=int, default=1000, help='lamb switch to adamw')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')
parser.add_argument('--power', type=int, default=1, help='power')

# Augmentation & regularization parameters ONLY FOR IMAGE NET
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--newton-maxiter', default=20, type=int,
                    help='max iterration in newton method')
parser.add_argument('--reset-drop', action='store_true', default=False,
                    help='whether to reset drop')
parser.add_argument('--kernel-method', type=str, default='cuda', choices=['torch', 'cuda'],
                    help='The implementation way of gaussian kernel method, choose from "cuda" and "torch"')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between node after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of inputs bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='/home/hexiang/TransferLearning_For_DVS/Results_new_refined/', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--device', type=int, default=0)

# Spike parameters
parser.add_argument('--step', type=int, default=10, help='Simulation time step (default: 10)')
parser.add_argument('--encode', type=str, default='direct', help='Input encode method (default: direct)')
parser.add_argument('--temporal-flatten', action='store_true',
                    help='Temporal flatten to channels. ONLY FOR EVENT DATA TRAINING BY ANN')
parser.add_argument('--adaptive-node', action='store_true')
parser.add_argument('--critical-loss', action='store_true')

# neuron type
parser.add_argument('--node-type', type=str, default='LIFNode', help='Node type in network (default: PLIF)')
parser.add_argument('--act-fun', type=str, default='GateGrad',
                    help='Surogate Function in node. Only for Surrogate nodes (default: AtanGrad)')
parser.add_argument('--threshold', type=float, default=.5, help='Firing threshold (default: 0.5)')
parser.add_argument('--tau', type=float, default=2., help='Attenuation coefficient (default: 2.)')
parser.add_argument('--requires-thres-grad', action='store_true')
parser.add_argument('--sigmoid-thres', action='store_true')

parser.add_argument('--loss-fn', type=str, default='ce', help='loss function (default: ce)')
parser.add_argument('--noisy-grad', type=float, default=0.,
                    help='Add noise to backward, sometime will make higher accuracy (default: 0.)')
parser.add_argument('--spike-output', action='store_true', default=False,
                    help='Using mem output or spike output (default: False)')
parser.add_argument('--n_groups', type=int, default=1)

# EventData Augmentation
parser.add_argument('--mix-up', action='store_true', help='Mix-up for event data (default: False)')
parser.add_argument('--cut-mix', action='store_true', help='CutMix for event data (default: False)')
parser.add_argument('--event-mix', action='store_true', help='EventMix for event data (default: False)')
parser.add_argument('--cutmix_beta', type=float, default=1.0, help='cutmix_beta (default: 1.)')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix_prib for event data (default: .5)')
parser.add_argument('--cutmix_num', type=int, default=1, help='cutmix_num for event data (default: 1)')
parser.add_argument('--cutmix_noise', type=float, default=0.,
                    help='Add Pepper noise after mix, sometimes work (default: 0.)')
parser.add_argument('--gaussian-n', type=int, default=3)
parser.add_argument('--rand-aug', action='store_true',
                    help='Rand Augment for Event data (default: False)')
parser.add_argument('--randaug_n', type=int, default=3,
                    help='Rand Augment times n (default: 3)')
parser.add_argument('--randaug_m', type=int, default=15,
                    help='Rand Augment times n (default: 15) (0-30)')
parser.add_argument('--train-portion', type=float, default=0.9,
                    help='Dataset portion, only for datasets which do not have validation set (default: 0.9)')
parser.add_argument('--event-size', default=48, type=int,
                    help='Event size. Resize event data before process (default: 48)')
parser.add_argument('--layer-by-layer', action='store_true',
                    help='forward step-by-step or layer-by-layer. '
                         'Larger Model with layer-by-layer will be faster (default: False)')
parser.add_argument('--node-resume', type=str, default='',
                    help='resume weights in node for adaptive node. (default: False)')
parser.add_argument('--node-trainable', action='store_true')

# visualize
parser.add_argument('--visualize', action='store_true',
                    help='Visualize spiking map for each layer, only for validate (default: False)')
parser.add_argument('--spike-rate', action='store_true',
                    help='Print spiking rate for each layer, only for validate(default: False)')
parser.add_argument('--tsne', action='store_true')
parser.add_argument('--conf-mat', action='store_true')

parser.add_argument('--suffix', type=str, default='',
                    help='Add an additional suffix to the save path (default: \'\')')

parser.add_argument('--DVS-DA', action='store_true',
                    help='use DA on DVS')

# train data used ratio
parser.add_argument('--traindata-ratio', default=1.0, type=float,
                    help='training data ratio')

# snr value
parser.add_argument('--snr', default=0, type=int,
                    help='random noise amplitude controled by snr, 0 means no noise')

parser.add_argument('--aug_smooth', action='store_true',
                    help='Apply test time augmentation to smooth the CAM')
parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet'
         'of cam_weights*activations')

import os
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from tonic.datasets import NCALTECH101, CIFAR10DVS
import tonic
from matplotlib import rcParams
import seaborn as sns


# for matplotlib 3D
def get_proj(self):
    """
     Create the projection matrix from the current viewing position.

     elev stores the elevation angle in the z plane
     azim stores the azimuth angle in the (x, y) plane

     dist is the distance of the eye viewing point from the object point.
    """
    # chosen for similarity with the initial view before gh-8896

    relev, razim = np.pi * self.elev / 180, np.pi * self.azim / 180

    # EDITED TO HAVE SCALED AXIS
    xmin, xmax = np.divide(self.get_xlim3d(), self.pbaspect[0])
    ymin, ymax = np.divide(self.get_ylim3d(), self.pbaspect[1])
    zmin, zmax = np.divide(self.get_zlim3d(), self.pbaspect[2])

    # transform to uniform world coordinates 0-1, 0-1, 0-1
    worldM = proj3d.world_transformation(xmin, xmax,
                                         ymin, ymax,
                                         zmin, zmax)

    # look into the middle of the new coordinates
    R = self.pbaspect / 2

    xp = R[0] + np.cos(razim) * np.cos(relev) * self.dist
    yp = R[1] + np.sin(razim) * np.cos(relev) * self.dist
    zp = R[2] + np.sin(relev) * self.dist
    E = np.array((xp, yp, zp))

    self.eye = E
    self.vvec = R - E
    self.vvec = self.vvec / np.linalg.norm(self.vvec)

    if abs(relev) > np.pi / 2:
        # upside down
        V = np.array((0, 0, -1))
    else:
        V = np.array((0, 0, 1))
    zfront, zback = -self.dist, self.dist

    viewM = proj3d.view_transformation(E, R, V)
    projM = self._projection(zfront, zback)
    M0 = np.dot(viewM, worldM)
    M = np.dot(projM, M0)
    return M


def event_vis_raw(x):
    sns.set_style('whitegrid')
    # sns.set_palette('deep', desat=.6)
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    Axes3D.get_proj = get_proj
    x = np.array(x.tolist())  # x, y, t, p
    mask = (x[:, 3] == 1)
    x_pos = x[mask]
    x_neg = x[mask == False]
    pos_idx = np.random.choice(x_pos.shape[0], 10000)
    neg_idx = np.random.choice(x_neg.shape[0], 10000)
    # x_pos[pos_idx, 2] = 0
    # x_neg[neg_idx, 2] = 0

    fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
    ax = Axes3D(fig)
    ax.pbaspect = np.array([2.0, 1.0, 0.5])
    ax.view_init(elev=10, azim=-75)
    # ax.view_init(elev=15, azim=15)
    ax.set_xlabel('t (time step)')
    ax.set_ylabel('w (pixel)')
    ax.set_zlabel('h (pixel)')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.scatter(x_pos[pos_idx, 2], 48 - x_pos[pos_idx, 0], 48 - x_pos[pos_idx, 1], color='red', alpha=0.3, s=1.)
    # ax.scatter(x_neg[neg_idx, 2], 48 - x_neg[neg_idx, 0], 48 - x_neg[neg_idx, 1], color='blue', alpha=0.3, s=1.)
    ax.scatter(x_pos[:, 0], 48 - x_pos[:, 1] * 0.375, 48 - x_pos[:, 2] * 0.375, color='red', alpha=0.3, s=1.)
    # ax.scatter(x_neg[:, 0], 64 - x_neg[:, 1] // 2, 128 - x_neg[:, 2], color='blue', alpha=0.3, s=1.)
    ax.scatter(18000, 48 - x_pos[:, 1] * 0.375, 48 - x_pos[:, 2] * 0.375, color='red', alpha=0.3, s=1.)
    # ax.scatter(18000, 64 - x_pos[:, 1] // 2, 128 - x_pos[:, 2], color='blue', alpha=0.3, s=1.)


def get_dataloader_ncal(step, **kwargs):
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size
    transform = tonic.transforms.Compose([
        # tonic.transforms.DropPixel(hot_pixel_frequency=.999),
        # tonic.transforms.Denoise(500),
        tonic.transforms.DropEvent(p=0.0),
        # tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=step),
        # lambda x: F.interpolate(torch.tensor(x, dtype=torch.float), size=[48, 48], mode='bilinear', align_corners=True),
    ])
    dataset = tonic.datasets.CIFAR10DVS(os.path.join(DATA_DIR, 'DVS/DVS_Cifar10'), transform=transform)
    # dataset = [dataset[5569], dataset[8196]]
    # dataset = [dataset[5000], dataset[6000]] # 1958
    # dataset = [dataset[0]]
    # loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=1,
    #     shuffle=False,
    #     pin_memory=True, drop_last=True, num_workers=8
    # )
    return dataset

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    torch.set_num_threads(20)
    os.environ["OMP_NUM_THREADS"] = "20"  # 设置OpenMP计算库的线程数
    os.environ["MKL_NUM_THREADS"] = "20"  # 设置MKL-DNN CPU加速库的线程数。
    args, args_text = _parse_args()
    args.no_spike_output = True
    torch.cuda.set_device('cuda:%d' % args.device)
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            _logger.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    # args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank

    assert args.rank >= 0

    if args.distributed:
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on %d GPUs.' % args.num_gpu)

    # torch.manual_seed(args.seed + args.rank)
    setup_seed(args.seed + args.rank)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        adaptive_node=args.adaptive_node,
        dataset=args.target_dataset,
        step=args.step,
        encode_type=args.encode,
        node_type=eval(args.node_type),
        threshold=args.threshold,
        tau=args.tau,
        sigmoid_thres=args.sigmoid_thres,
        requires_thres_grad=args.requires_thres_grad,
        spike_output=not args.no_spike_output,
        act_fun=args.act_fun,
        temporal_flatten=args.temporal_flatten,
        layer_by_layer=args.layer_by_layer,
        n_groups=args.n_groups,
    )

    if 'dvs' in args.target_dataset:
        args.channels = 2
    elif 'mnist' in args.target_dataset:
        args.channels = 1
    else:
        args.channels = 3
    # flops, params = profile(model, inputs=(torch.randn(1, args.channels, args.event_size, args.event_size),), verbose=False)
    # _logger.info('flops = %fM', flops / 1e6)
    # _logger.info('param size = %fM', params / 1e6)

    linear_scaled_lr = args.lr * args.batch_size * args.world_size / 1024.0
    args.lr = linear_scaled_lr
    _logger.info("learning rate is %f" % linear_scaled_lr)

    if args.local_rank == 0:
        _logger.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    # now config only for imnet
    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    # source_loader_train, _, _, _ = eval('get_transfer_%s_data' % args.source_dataset)(
    #     batch_size=args.batch_size,
    #     step=args.step,
    #     args=args,
    #     _logge=_logger,
    #     data_config=data_config,
    #     size=args.event_size,
    #     mix_up=args.mix_up,
    #     cut_mix=args.cut_mix,
    #     event_mix=args.event_mix,
    #     beta=args.cutmix_beta,
    #     prob=args.cutmix_prob,
    #     gaussian_n=args.gaussian_n,
    #     num=args.cutmix_num,
    #     noise=args.cutmix_noise,
    #     num_classes=args.num_classes,
    #     rand_aug=args.rand_aug,
    #     randaug_n=args.randaug_n,
    #     randaug_m=args.randaug_m,
    #     portion=args.train_portion,
    #     _logger=_logger,
    # )


    origin_loader_train, _, _, _ = eval('get_origin_dvsc10_data')(
        batch_size=args.batch_size,
        step=args.step,
        args=args,
        _logge=_logger,
        data_config=data_config,
        size=args.event_size,
        mix_up=args.mix_up,
        cut_mix=args.cut_mix,
        event_mix=args.event_mix,
        beta=args.cutmix_beta,
        prob=args.cutmix_prob,
        gaussian_n=args.gaussian_n,
        num=args.cutmix_num,
        noise=args.cutmix_noise,
        num_classes=args.num_classes,
        rand_aug=args.rand_aug,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
        portion=args.train_portion,
        _logger=_logger,
    )

    target_loader_train, target_loader_eval, mixup_active, mixup_fn = eval('get_%s_data' % args.target_dataset)(
        batch_size=args.batch_size,
        dvs_da=args.DVS_DA,
        step=args.step,
        args=args,
        _logge=_logger,
        data_config=data_config,
        size=args.event_size,
        mix_up=args.mix_up,
        cut_mix=args.cut_mix,
        event_mix=args.event_mix,
        beta=args.cutmix_beta,
        prob=args.cutmix_prob,
        gaussian_n=args.gaussian_n,
        num=args.cutmix_num,
        noise=args.cutmix_noise,
        num_classes=args.num_classes,
        rand_aug=args.rand_aug,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
        portion=args.train_portion,
        _logger=_logger,
        train_data_ratio=args.traindata_ratio,
        snr=args.snr,
        data_mode="full",
        frames_num=12,
        data_type="frequency"
    )

    model_before = deepcopy(model)
    if args.eval:  # evaluate the model
        if args.distributed:
            state_dict = torch.load(args.eval_checkpoint)['state_dict_ema']
            new_state_dict = OrderedDict()
            # add module prefix for DDP
            for k, v in state_dict.items():
                k = 'module.' + k
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(torch.load('/home/hexiang/TransferLearning_For_DVS/Results_lastest/train_TCKA_test/Transfer_VGG_SNN-dvsc10-10-bs_120-seed_42-DA_True-ls_0.0-lr_0.005-SNR_0-domainLoss_False-semanticLoss_False-domain_loss_coefficient1.0-semantic_loss_coefficient0.5-traindataratio_1.0-TETfirst_True-TETsecond_True/model_best.pth.tar', map_location='cpu')['state_dict'])
            # pass
            # print("no model load")
        # --------------------------------------------------------------------------
        # Show Acc
        # --------------------------------------------------------------------------
        print("load model finished!")


    # """ python cam.py -image-path <path_to_image>
    # Example usage of loading an image, and computing:
    #     1. CAM
    #     2. Guided Back Propagation
    #     3. Combining both
    # """
    #
    # # Choose the target layer you want to compute the visualization for.
    # # Usually this will be the last convolutional layer in the model.
    # # Some common choices can be:
    # # Resnet18 and 50: model.layer4
    # # VGG, densenet161: model.features[-1]
    # # mnasnet1_0: model.layers[-1]
    # # You can print the model to help chose the layer
    # # You can pass a list with several target layers,
    # # in that case the CAMs will be computed per layer and then aggregated.
    # # You can also try selecting all layers of a certain type, with e.g:
    # # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # # find_layer_types_recursive(model, [torch.nn.ReLU])
    # target_layers = [model.feature[-1]]
    #
    # if True:
    #     # inputs = 0.0
    #     # label = 0.0
    #     # for batch_idx, (inputs_tmp, label_tmp) in tqdm.tqdm(enumerate(origin_loader_train)):
    #     #     if batch_idx == choose_idx:
    #     #         inputs = inputs_tmp
    #     #         label = label_tmp
    #     #         break
    #     #     else:
    #     #         continue
    #     inputs = 0.0
    #     rgb_img = 0.0
    #
    #     #Using the with statement ensures the context is freed, and you can
    #     #recreate different CAM objects in a loop.
    #     plt.figure(figsize=(8, 6))
    #     plt.xlabel('w (pixel)')
    #     plt.ylabel('h (pixel)')
    #     cam_algorithm = GradCAMPlusPlus
    #     model = model.cuda()
    #     with cam_algorithm(model=model,
    #                        target_layers=target_layers,
    #                        use_cuda=False) as cam:
    #
    #         # AblationCAM and ScoreCAM have batched implementations.
    #         # You can override the internal batch size for faster computation.
    #         cam.batch_size = 32
    #
    #         for batch_idx, (origin_loaer, target_loader) in tqdm.tqdm(enumerate(zip(origin_loader_train, target_loader_train))):
    #
    #             twodemension_inputs, labels = origin_loaer
    #             plt.figure(figsize=(8, 6))
    #             # plt.xlabel('w (pixel)')
    #             # plt.ylabel('h (pixel)')
    #             twodemension_inputs = twodemension_inputs[0]  # (1, 10, 2, 48, 48) -> (10, 2, 48, 48)
    #             event_frame_plot_2d(twodemension_inputs)
    #
    #             inputs_tmp, label_tmp = target_loader
    #             inputs = inputs_tmp
    #             inputs = inputs.type(torch.FloatTensor).cuda()
    #
    #             grayscale_cam = cam(input_tensor=inputs,
    #                                 targets=None,
    #                                 aug_smooth=args.aug_smooth,
    #                                 eigen_smooth=args.eigen_smooth)
    #
    #             # Here grayscale_cam has only one image in the batch
    #             grayscale_cam = grayscale_cam[0, :]
    #
    #             # cam_image = show_cam_on_image(rgb_img.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True, image_weight=0.0)
    #             cam_image = show_cam_on_image(np.ones((48, 48, 3)), grayscale_cam, use_rgb=True,
    #                                           image_weight=0.0)
    #         # # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    #     # rgb_img = cv2.resize(rgb_img.permute(1, 2, 0).numpy(), (32, 32))
    #
    #             # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    #             plt.ylim(bottom=0.)
    #             plt.axis('off')
    #             plt.savefig('fig/gradcam_dvspic_origin/label_{}_id_{}.jpg'.format(labels.item(), 400 + batch_idx), bbox_inches='tight', pad_inches=0)
    #             plt.imshow(cam_image, alpha=1.0)
    #             # plt.show()
    #             # plt.savefig('gradcam_pic/plot_id{}.jpg'.format(batch_idx), bbox_inches='tight')
    #             plt.savefig('fig/gradcam_dvspic_withoutloss/label_{}_id_{}.jpg'.format(labels.item(), 400 + batch_idx), bbox_inches='tight', pad_inches=0)

    # 第二次
    print("load model again!")
    model = model_before
    model.load_state_dict(torch.load(
        '/home/hexiang/TransferLearning_For_DVS/Results_lastest/train_TCKA_test/Transfer_VGG_SNN-dvsc10-10-bs_120-seed_42-DA_True-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.5-traindataratio_1.0-TETfirst_True-TETsecond_True/model_best.pth.tar', map_location='cpu')['state_dict'])


    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.feature[-1]]

    if True:
        # inputs = 0.0
        # label = 0.0
        # for batch_idx, (inputs_tmp, label_tmp) in tqdm.tqdm(enumerate(origin_loader_train)):
        #     if batch_idx == choose_idx:
        #         inputs = inputs_tmp
        #         label = label_tmp
        #         break
        #     else:
        #         continue
        inputs = 0.0
        rgb_img = 0.0

        #Using the with statement ensures the context is freed, and you can
        #recreate different CAM objects in a loop.
        plt.figure(figsize=(8, 6))
        plt.xlabel('w (pixel)')
        plt.ylabel('h (pixel)')
        cam_algorithm = GradCAMPlusPlus
        model = model.cuda()
        with cam_algorithm(model=model,
                           target_layers=target_layers,
                           use_cuda=False) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32

            for batch_idx, (origin_loaer, target_loader) in tqdm.tqdm(enumerate(zip(origin_loader_train, target_loader_train))):

                twodemension_inputs, labels = origin_loaer
                plt.figure(figsize=(8, 6))
                # plt.xlabel('w (pixel)')
                # plt.ylabel('h (pixel)')
                twodemension_inputs = twodemension_inputs[0]  # (1, 10, 2, 48, 48) -> (10, 2, 48, 48)
                event_frame_plot_2d(twodemension_inputs)

                inputs_tmp, label_tmp = target_loader
                inputs = inputs_tmp
                inputs = inputs.type(torch.FloatTensor).cuda()

                grayscale_cam = cam(input_tensor=inputs,
                                    targets=None,
                                    aug_smooth=args.aug_smooth,
                                    eigen_smooth=args.eigen_smooth)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]

                # cam_image = show_cam_on_image(rgb_img.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True, image_weight=0.0)
                cam_image = show_cam_on_image(np.ones((48, 48, 3)), grayscale_cam, use_rgb=True,
                                              image_weight=0.0)
            # # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        # rgb_img = cv2.resize(rgb_img.permute(1, 2, 0).numpy(), (32, 32))

                # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
                plt.ylim(bottom=0.)
                plt.axis('off')
                # plt.savefig('fig/gradcam_dvspic_origin/label_{}_id_{}.jpg'.format(labels.item(), batch_idx), bbox_inches='tight', pad_inches=0)
                plt.imshow(cam_image, alpha=1.0)
                # plt.show()
                # plt.savefig('gradcam_pic/plot_id{}.jpg'.format(batch_idx), bbox_inches='tight')
                plt.savefig('fig/gradcam_dvspic_withloss/label_{}_id_{}.jpg'.format(labels.item(), batch_idx), bbox_inches='tight', pad_inches=0)

def event_frame_plot_2d(event):

    for t in range(event.shape[0]):
        pos_idx = []
        neg_idx = []
        for x in range(event.shape[2]):
            for y in range(event.shape[3]):
                if event[t, 0, x, y] > 0:
                    pos_idx.append((x, y, event[t, 0, x, y]))
                if event[t, 1, x, y] > 0:
                    neg_idx.append((x, y, event[t, 0, x, y]))
        if len(pos_idx) > 0:
            # print(t)
            pos_x, pos_y, pos_c = np.split(np.array(pos_idx), 3, axis=1)
            # plt.scatter(48 - pos_x[:, 0] * 0.375, 48 - pos_y[:, 0] * 0.375, c='red', alpha=1, s=1)
            plt.scatter(pos_x[:, 0] * 0.375, pos_y[:, 0] * 0.375, c='white', alpha=1, s=1)
        if len(neg_idx) > 0:
            neg_x, neg_y, neg_c = np.split(np.array(neg_idx), 3, axis=1)
            # plt.scatter(48 - neg_x[:, 0] * 0.375, 48 - neg_y[:, 0] * 0.375, c='blue', alpha=1, s=1)
            plt.scatter(neg_x[:, 0] * 0.375, neg_y[:, 0] * 0.375, c='blue', alpha=1, s=1)
    # sys.exit()

if __name__ == '__main__':
    main()