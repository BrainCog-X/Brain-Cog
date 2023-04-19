# -*- coding: utf-8 -*-            
# Time : 2022/9/29 15:27
# Author : Regulus
# FileName: main_transfer.py
# Explain:
# Software: PyCharm

import argparse
import math
import time
import CKA
import numpy
import timm.models
import random as rd
import yaml
import os
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

# from ptflops import get_model_complexity_info
# from thop import profile, clever_format

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
parser.add_argument('--output', default='/home/hexiang/TransferLearning_For_DVS/Results_lastest/', type=str, metavar='PATH',
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

# Transfer Learning loss choice
parser.add_argument('--domain-loss', action='store_true',
                    help='add domain loss')
parser.add_argument('--semantic-loss', action='store_true',
                    help='add semantic loss')
parser.add_argument('--domain-loss-coefficient', type=float, default=1.0,
                    help='domain loss coefficient(default: 1.0)')
parser.add_argument('--semantic-loss-coefficient', type=float, default=1.0,
                    help='domain loss coefficient(default: 1.0)')

# use TET loss or not (all default False, do not use)

parser.add_argument('--TET-loss-first', action='store_true',
                    help='use TET loss one part')

parser.add_argument('--TET-loss-second', action='store_true',
                    help='use TET loss two part')

parser.add_argument('--DVS-DA', action='store_true',
                    help='use DA on DVS')

# train data used ratio
parser.add_argument('--traindata-ratio', default=1.0, type=float,
                    help='training data ratio')

# snr value
parser.add_argument('--snr', default=0, type=int,
                    help='random noise amplitude controled by snr, 0 means no noise')

# margin m
parser.add_argument('--m', default=-1.0, type=float,
                    help='margin')

source_input_list, source_label_list = [], []
CALTECH101_list, ImageNet_list = [], []

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
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = '-'.join([
            args.model,
            args.target_dataset,
            str(args.step),
            "bs_{}".format(args.batch_size),
            "seed_{}".format(args.seed),
            "DA_{}".format(args.DVS_DA),
            "ls_{}".format(args.smoothing),
            "lr_{}".format(args.lr),
            "m_{}".format(args.m),
            "domainLoss_{}".format(args.domain_loss),
            "semanticLoss_{}".format(args.semantic_loss),
            "domain_loss_coefficient{}".format(args.domain_loss_coefficient),
            "semantic_loss_coefficient{}".format(args.semantic_loss_coefficient),
            "traindataratio_{}".format(args.traindata_ratio),
            "TETfirst_{}".format(args.TET_loss_first),
            "TETsecond_{}".format(args.TET_loss_second),
        ])
        output_dir = get_outdir(output_base, 'train_TCKA_test_nop', exp_name)
        args.output_dir = output_dir
        setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))

    else:
        setup_default_logging()

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
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    else:
        torch.cuda.set_device('cuda:%d' % args.device)
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


    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            _logger.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model = model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer(args, model)

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume and args.eval_checkpoint == '':
        args.eval_checkpoint = args.resume
    if args.resume:
        args.eval = True
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    if args.critical_loss or args.spike_rate:
        model.set_requires_fp(True)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.node_resume:
        ckpt = torch.load(args.node_resume, map_location='cpu')
        model.load_node_weight(ckpt, args.node_trainable)

    model_without_ddp = model
    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex and use_amp != 'native':
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    _logger.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                _logger.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank],
                              find_unused_parameters=True)  # can use device str in Torch >= 1.1
        model_without_ddp = model.module
    # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # now config only for imnet
    data_config = resolve_data_config(vars(args), model=model, verbose=False)
    source_loader_train, _, _, _ = eval('get_transfer_%s_data' % args.source_dataset)(
        batch_size=args.batch_size,
        step=args.step,
        args=args,
        _logge=_logger,
        data_config=data_config,
        num_aug_splits=num_aug_splits,
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
        num_aug_splits=num_aug_splits,
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

    global source_input_list, source_label_list, CALTECH101_list, ImageNet_list
    if args.target_dataset == "dvsc10" or args.target_dataset == "NCALTECH101" or args.target_dataset == "nomni":  # ImageNet中回来的loader其实是数据集,在后面处理
        source_input_list, source_label_list = next(iter(source_loader_train))
        # for i in range(30001, 30005):
        #     # vis origin picture
        #     plt.figure()
        #     plt.imshow(source_input_list[i].permute(1, 2, 0).numpy())
        #     plt.savefig("./origin_image.jpg")
        #     plt.show()
        # vis HSV picture
        # for i in range(30001, 30005):  # 30001.i
        #     convertor = RGB_HSV()
        #     plt.figure()
        #     plt.imshow(convertor.rgb_to_hsv(source_input_list)[i, :, :, :].permute(1, 2, 0).numpy())
        #     plt.title("HSV image")
        #     plt.show()

    if args.source_dataset == "CALTECH101":
        cls_count = [438, 435, 200, 791, 49, 800, 41, 34, 45, 50, 45, 32, 128, 84, 38, 81, 86, 47, 40, 0, 45, 58, 61,
                     105, 47, 64, 70, 68, 50, 51, 54, 67, 51, 64, 65, 72, 62, 52, 60, 83, 65, 67, 45, 31, 34, 49, 99,
                     100, 42, 54, 86, 80, 30, 62, 86, 110, 61, 79, 77, 40, 65, 42, 35, 77, 31, 74, 49, 32, 39, 47, 35,
                     43, 52, 34, 54, 69, 58, 45, 38, 57, 34, 84, 57, 31, 54, 45, 82, 56, 63, 35, 85, 43, 82, 74, 239,
                     37, 53, 33, 55, 29, 42]
        CALTECH101_list = [0] * 102  # 多开了一类, 方便计算
        for i in range(1, len(cls_count) + 1):
            CALTECH101_list[i] = CALTECH101_list[i - 1] + cls_count[i - 1]

    if args.source_dataset == "NCALTECH101":
        cls_count = tonic.datasets.NCALTECH101.cls_count
        CALTECH101_list = [0] * 102  # 多开了一类, 方便计算
        for i in range(1, len(cls_count) + 1):
            CALTECH101_list[i] = CALTECH101_list[i - 1] + cls_count[i - 1]

    if args.source_dataset == "imnet":
        cls_count = [1300] * 1000  # 1000类
        cls_count_idx = [1117, 1266, 1071, 1141, 1272, 1150, 772, 860, 1136, 732, 1025, 754, 1290, 738, 1258, 1273, 977,
                         936, 1156, 1218, 969, 954, 1070, 755, 1206, 1165, 969, 1292, 1236, 1199, 1209, 1176, 1186,
                         1194,
                         1067, 1029, 1154, 1216, 1187, 889, 1211, 1136, 1153, 1222, 1282, 1283, 980, 1034, 891, 1285,
                         986,
                         1137, 1272, 1155, 1097, 1149, 1155, 1159, 1133, 1180, 1120, 1005, 1152, 1156, 962, 1157, 1282,
                         1117, 1118, 1270, 1069, 1053, 1254, 908, 1247, 1253, 1029, 1259, 1267, 1249, 1162, 1045, 1004,
                         1238, 1153, 1084, 1217, 931, 1264, 976, 1250, 1053, 1160, 1062, 1137, 1299, 1055, 1213, 1206,
                         1154,
                         1207, 1149, 1239, 1125, 1193]
        cls_idx = [43, 51, 62, 98, 103, 147, 152, 158, 164, 165, 166, 167, 168, 175, 181, 183, 188, 190, 194, 206, 221,
                   252, 262, 268, 335, 390, 392, 409, 418, 426, 439, 465, 481, 491, 499, 501, 503, 507, 521, 531, 536,
                   550, 551, 567, 577, 583, 585, 590, 596, 610, 623, 630, 631, 635, 653, 662, 663, 675, 676, 678, 686,
                   689, 706, 708, 712, 714, 722, 723, 724, 727, 728, 729, 731, 740, 747, 753, 771, 772, 782, 789, 798,
                   810, 811, 812, 821, 826, 838, 841, 854, 857, 860, 869, 872, 885, 891, 892, 901, 906, 914, 921, 925,
                   926, 940, 946, 969]
        for i in range(len(cls_count)):
            if i in cls_idx:
                cls_count[i] = cls_count_idx[cls_idx.index(i)]
        ImageNet_list = [0] * 1001  # 多开了一类, 方便计算
        for i in range(1, 1000 + 1):
            ImageNet_list[i] = ImageNet_list[i - 1] + cls_count[i - 1]


    if args.loss_fn == 'mse':
        train_loss_fn = UnilateralMse(1.)
        validate_loss_fn = UnilateralMse(1.)

    else:
        if args.jsd:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
        elif mixup_active:
            # smoothing is handled with mixup target transform
            train_loss_fn = SoftTargetCrossEntropy().cuda()
        elif args.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        else:
            train_loss_fn = nn.CrossEntropyLoss().cuda()

        validate_loss_fn = nn.CrossEntropyLoss().cuda()


    if args.loss_fn == 'mix':
        train_loss_fn = MixLoss(train_loss_fn)
        validate_loss_fn = MixLoss(validate_loss_fn)

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None


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
            model.load_state_dict(torch.load(args.eval_checkpoint)['state_dict'])
        for i in range(1):
            val_metrics = validate(start_epoch, model, target_loader_eval, validate_loss_fn, args,
                                   visualize=args.visualize, spike_rate=args.spike_rate,
                                   tsne=args.tsne, conf_mat=args.conf_mat)
            print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
        return

    saver = None
    if args.local_rank == 0:
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=1)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    eval_top1 = 0.0
    try:  # train the model
        if args.reset_drop:
            model_without_ddp.reset_drop_path(0.0)
        for epoch in range(start_epoch, args.epochs):
            if epoch == 0 and args.reset_drop:
                model_without_ddp.reset_drop_path(args.drop_path)

            if args.distributed:
                target_loader_train.sampler.set_epoch(epoch)
            train_metrics = train_epoch(
                epoch, model, source_loader_train, target_loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(epoch, model, target_loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                                    visualize=args.visualize, spike_rate=args.spike_rate,
                                    tsne=args.tsne, conf_mat=args.conf_mat)
            eval_top1 = eval_metrics["top1"]

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                    ema_eval_metrics = validate(
                        epoch, model_ema.ema, target_loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)',
                        visualize=args.visualize, spike_rate=args.spike_rate,
                        tsne=args.tsne, conf_mat=args.conf_mat)
                    eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

            # if saver is not None and epoch >= args.n_warm_up:
            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            # if epoch == 299:  # 临时的
            #     break

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(
        epoch, model, source_loader, target_loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and target_loader.mixup_enabled:
            target_loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    domain_losses_m = AverageMeter()
    semantic_losses_m = AverageMeter()
    rgb_losses_m = AverageMeter()
    dvs_losses_m = AverageMeter()
    closses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(target_loader) - 1
    num_updates = epoch * len(target_loader)
    convertor = RGB_HSV()

    batch_len = len(target_loader)
    if args.target_dataset == "dvsc10":
        set_MaxReplacement_epoch = 0.5 * args.epochs
    else:
        set_MaxReplacement_epoch = 0.5 * args.epochs
    P_Replacement = 0.0

    global source_input_list, source_label_list, CALTECH101_list, ImageNet_list
    for batch_idx, (inputs, label) in enumerate(target_loader):
        P_Replacement = ((batch_idx + epoch * batch_len) / (set_MaxReplacement_epoch * batch_len)) ** 3
        P_Replacement = P_Replacement if P_Replacement <= 1.0 else 1.0
        sampler_list = label.tolist()
        if args.target_dataset == "dvsc10" and args.source_dataset == "cifar10":
            sampler_list = torch.tensor(sampler_list) * 6000 + torch.randint(0, 6000, (len(sampler_list),))
        elif args.target_dataset == "dvsc10" and args.source_dataset == "dvsc10":
            sampler_list = torch.tensor(sampler_list) * 900 + torch.randint(0, 900, (len(sampler_list),))
        elif args.target_dataset == "NCALTECH101":
            tmp_sampler_list = []
            idx_list = []
            for idx, label_sampler in enumerate(sampler_list):
                if label_sampler == 19:
                    tmp_sampler_list.append(0)
                    idx_list.append(idx)
                else:
                    tmp_sampler_list.append(torch.randint(CALTECH101_list[label_sampler],
                                                          CALTECH101_list[label_sampler + 1], (1,)).item())
        elif args.target_dataset == "esimnet":
            tmp_sampler_list = []
            for idx, label_sampler in enumerate(sampler_list):  # 这里的label_sampler是一个列表
                tmp_sampler_list.append(torch.randint(ImageNet_list[label_sampler],
                                                      ImageNet_list[label_sampler + 1], (1,)).item())
        elif args.target_dataset == "nomni":
            sampler_list = torch.tensor(sampler_list) * 20 + torch.randint(0, 20, (len(sampler_list),))

        source_input, source_label = [], []
        if args.target_dataset == "dvsc10":
            source_input, source_label = source_input_list[sampler_list], source_label_list[sampler_list]
        if args.target_dataset == "NCALTECH101":
            source_input, source_label = source_input_list[tmp_sampler_list], source_label_list[tmp_sampler_list]
        if args.target_dataset == "esimnet":
            train_dataset = source_loader  # 给传回来的重新命个名儿
            source_loader_used = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size, shuffle=False,
                num_workers=8, pin_memory=True, sampler=TransferSampler(tmp_sampler_list))
            source_input, source_label = next(iter(source_loader_used))
        if args.target_dataset == "nomni":
            source_input, source_label = source_input_list[sampler_list], source_label_list[sampler_list]
        # for i in range(128):
        #     # vis origin picture
        #     plt.figure()
        #     plt.imshow(source_input[i].permute(1, 2, 0))
        #     plt.title("origin image")
        #     plt.show()

        # # vis HSV picture
        # plt.figure()
        # plt.imshow(convertor.rgb_to_hsv(inputs)[7, :, :, :].permute(1, 2, 0).numpy())
        # plt.title("HSV image")
        # plt.show()

        # source_input = convertor.rgb_to_hsv(source_input)[:, -1, :, :].unsqueeze(1).repeat(1, args.step * 2, 1, 1)
        if args.source_dataset == "dvsc10" or args.source_dataset == "NCALTECH101":
            pass
        else:
            source_input = source_input[:, -1, :, :].unsqueeze(1).repeat(1, args.step * 2, 1, 1)
            source_input = rearrange(source_input, 'b (t c) h w -> b t c h w', t=args.step)

        for b in range(source_input.shape[0]):
            if rd.uniform(0, 1) <= P_Replacement:
                source_input[b] = inputs[b, :, :, :, :]

        # for i in range(10):
        #     # vis HSV picture for v channel
        #     plt.figure()
        #     plt.imshow(source_input[i][0].permute(1, 2, 0)[:, :, -1].unsqueeze(2))
        #     plt.title("HSV image for v channel")
        #     plt.show()

        if args.target_dataset == "NCALTECH101" and len(idx_list) > 0:
            for i in range(len(idx_list)):
                source_input[idx_list[i]] = inputs[idx_list[i]]
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher or args.target_dataset != 'imnet':
            inputs, label = inputs.type(torch.FloatTensor).cuda(), label.cuda()
            source_input, source_label = source_input.type(torch.FloatTensor).cuda(), label.cuda()
            if mixup_fn is not None:
                inputs, label = mixup_fn(inputs, label)
                source_input, source_label = mixup_fn(source_input, source_label)
        if args.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
            source_input = source_input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            domain_rbg_list, domain_dvs_list, output_rgb, output_dvs = model(source_input, inputs)

            # compute semantic loss
            label_idx = [[] for i in range(args.num_classes)]
            semantic_label_list = []
            for idx, i in enumerate(label):
                label_idx[i.item()].append(idx)
            for i in label:
                while True:
                    label_tmp = torch.randint(0, args.num_classes, (1,)).item()
                    if i.item() != label_tmp and len(label_idx[label_tmp]) > 0:  # NCALTECH101有空列表, 需要判断
                        break
                semantic_label_list.append(int(np.random.choice(label_idx[label_tmp], 1)))
            semantic_rbg_list = []
            semantic_loss = 0.
            for i in range(len(domain_rbg_list)):
                semantic_rbg_list.append(domain_rbg_list[i][semantic_label_list])
            for i in range(len(domain_rbg_list)):
                semantic_loss += torch.abs(CKA.linear_CKA(domain_dvs_list[i].view(args.batch_size, -1), semantic_rbg_list[i].view(args.batch_size, -1)))
            semantic_loss /= len(domain_rbg_list)
            if args.target_dataset == "dvsc10":
                m = 0.1
            elif args.target_dataset == "NCALTECH101":
                m = 0.3
            else:
                m = 0.2
            if args.m >= 0.0:
                m = args.m
            if semantic_loss.item() - m <= 0:
                semantic_loss = torch.tensor(0., device=semantic_loss.device)

            # if args.domain_loss_after:
            #     # compute domain loss
            #     for b in range(source_input.shape[0]):
            #         if rd.uniform(0, 1) <= P_Replacement:
            #             for i in range(len(domain_rbg_list)):
            #                 domain_rbg_list[i][b] = domain_dvs_list[i][b, :, :, :]

            domain_loss = 0.
            for i in range(len(domain_rbg_list)):
                domain_loss += 1 - torch.abs(CKA.linear_CKA(domain_rbg_list[i].view(args.batch_size, -1), domain_dvs_list[i].view(args.batch_size, -1)))
            domain_loss /= len(domain_rbg_list)

            # compute cls loss
            lamb = 1e-3
            if args.TET_loss_first or args.TET_loss_second:  # 第一项必须有，也就是测两个，第一个何第一个加第二个
                loss_rgb = 0
                tet_loss_first = 0
                tet_loss_second = 0
                assert len(output_rgb) == len(output_dvs)
                for i in range(len(output_rgb)):
                    loss_rgb += loss_fn(output_rgb[i], label)
                    tet_loss_first += loss_fn(output_dvs[i], label)
                loss_rgb /= len(output_rgb)
                tet_loss_first /= len(output_dvs)

                if args.TET_loss_second:
                    y = torch.zeros_like(output_dvs[-1]).fill_(args.threshold)
                    secondLoss = torch.nn.MSELoss()
                    tet_loss_second = secondLoss(output_dvs[-1], y)
                else:
                    lamb = 0.0
                loss_dvs = (1 - lamb) * tet_loss_first + lamb * tet_loss_second
                output_rgb = sum(output_rgb) / len(output_rgb)
                output_dvs = sum(output_dvs) / len(output_dvs)
            else:
                output_rgb = sum(output_rgb) / len(output_rgb)
                output_dvs = sum(output_dvs) / len(output_dvs)
                loss_rgb = loss_fn(output_rgb, label)
                loss_dvs = loss_fn(output_dvs, label)

            loss = 0 * loss_rgb + loss_dvs
            if args.domain_loss:
                loss += args.domain_loss_coefficient * domain_loss
            if args.semantic_loss and epoch <= set_MaxReplacement_epoch:
                if args.target_dataset == "NCALTECH101" and epoch <= set_MaxReplacement_epoch * 0.66:
                    # loss += args.semantic_loss_coefficient * semantic_loss * math.pow(10, -1.0 * float(set_MaxReplacement_epoch / (epoch+1)))
                    pass
                else:
                    loss += args.semantic_loss_coefficient * semantic_loss

        if not (args.cut_mix | args.mix_up | args.event_mix) and args.target_dataset != 'imnet':
            acc1, acc5 = accuracy(output_dvs, label, topk=(1, 5))
        else:
            acc1, acc5 = torch.tensor([0.]), torch.tensor([0.])

        closs = torch.tensor([0.], device=loss.device)

        loss = loss + .1 * closs

        spike_rate_avg_layer_str = ''
        threshold_str = ''
        if not args.distributed:
            losses_m.update(loss.item(), inputs.size(0))
            domain_losses_m.update(domain_loss.item(), inputs.size(0))
            semantic_losses_m.update(semantic_loss.item(), inputs.size(0))
            rgb_losses_m.update(loss_rgb.item(), inputs.size(0))
            dvs_losses_m.update(loss_dvs.item(), inputs.size(0))
            top1_m.update(acc1.item(), inputs.size(0))
            top5_m.update(acc5.item(), inputs.size(0))
            closses_m.update(closs.item(), inputs.size(0))

            spike_rate_avg_layer = model.get_fire_rate().tolist()
            spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in spike_rate_avg_layer]
            threshold = model.get_threshold()
            threshold_str = ['{:.3f}'.format(i.item()) for i in threshold]

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(), create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.noisy_grad != 0.:
                random_gradient(model, args.noisy_grad)
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            if args.opt == 'lamb':
                optimizer.step(epoch=epoch)
            else:
                optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            mu_str = ''
            sigma_str = ''
            if not args.distributed:
                if 'Noise' in args.node_type:
                    mu, sigma = model.get_noise_param()
                    mu_str = ['{:.3f}'.format(i.detach()) for i in mu]
                    sigma_str = ['{:.3f}'.format(i.detach()) for i in sigma]

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), inputs.size(0))
                closses_m.update(reduced_loss.item(), inputs.size(0))

            if args.local_rank == 0:
                if args.distributed:
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                        'cLoss: {closs.val:>9.6f} ({closs.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(target_loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            closs=closses_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m,
                            rate=inputs.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m
                        ))
                else:
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                        'cLoss: {closs.val:>9.6f} ({closs.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})\n'
                        'Fire_rate: {spike_rate}\n'
                        'Thres: {threshold}\n'
                        'Mu: {mu_str}\n'
                        'Sigma: {sigma_str}\n'
                        'P_Replacement: {P_Replacement}\n'.format(
                            epoch,
                            batch_idx, len(target_loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            closs=closses_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m,
                            rate=inputs.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=inputs.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m,
                            spike_rate=spike_rate_avg_layer_str,
                            threshold=threshold_str,
                            mu_str=mu_str,
                            sigma_str=sigma_str,
                            P_Replacement=P_Replacement,
                        ))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        inputs,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
    # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg), ('domainLoss', domain_losses_m.avg), ('semanticLoss', semantic_losses_m.avg),
                        ('rgbLoss', rgb_losses_m.avg), ('dvsLoss', dvs_losses_m.avg)])

def validate(epoch, model, loader, loss_fn, args, amp_autocast=suppress,
             log_suffix='', visualize=False, spike_rate=False, tsne=False, conf_mat=False):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    closses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    feature_vec = []
    feature_cls = []
    logits_vec = []
    labels_vec = []

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            # inputs = inputs.type(torch.float64)
            last_batch = batch_idx == last_idx
            if not args.prefetcher or args.target_dataset != 'imnet':
                inputs = inputs.type(torch.FloatTensor).cuda()
                target = target.cuda()
            if args.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            if not args.distributed:
                if (visualize or spike_rate or tsne or conf_mat) and not args.critical_loss:
                    model.set_requires_fp(True)
                    # if not args.critical_loss:
                    #     model.set_requires_fp(False)

            with amp_autocast():
                _, _, output_rbg, output_dvs = model(inputs, inputs)
                output = sum(output_dvs) / len(output_dvs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            if not args.distributed:
                if visualize:
                    x = model.get_fp()
                    feature_path = os.path.join(args.output_dir, 'feature_map')
                    if os.path.exists(feature_path) is False:
                        os.mkdir(feature_path)
                    save_feature_map(x, feature_path)
                    # if not args.critical_loss:
                    #     model_config.set_requires_fp(False)

                if tsne:
                    x = model.get_fp(temporal_info=False)[-1]
                    x = torch.nn.AdaptiveAvgPool2d((1, 1))(x)
                    x = x.reshape(x.shape[0], -1)
                    feature_vec.append(x)
                    feature_cls.append(target)

                if conf_mat:
                    logits_vec.append(output)
                    labels_vec.append(target)

                if spike_rate:
                    avg, var, spike, avg_per_step = model.get_spike_info()
                    save_spike_info(
                        os.path.join(args.output_dir, 'spike_info.csv'),
                        epoch, batch_idx,
                        args.step, avg, var,
                        spike, avg_per_step)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # acc1, = accuracy(output, target)

            closs = torch.tensor([0.], device=loss.device)

            if not args.distributed:
                spike_rate_avg_layer = model.get_fire_rate().tolist()
                threshold = model.get_threshold()
                threshold_str = ['{:.3f}'.format(i) for i in threshold]
                spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in spike_rate_avg_layer]
                tot_spike = model.get_tot_spike()

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            closses_m.update(closs.item(), inputs.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix

                mu_str = ''
                sigma_str = ''
                if not args.distributed:
                    if 'Noise' in args.node_type:
                        mu, sigma = model.get_noise_param()
                        mu_str = ['{:.3f}'.format(i.detach()) for i in mu]
                        sigma_str = ['{:.3f}'.format(i.detach()) for i in sigma]

                if args.distributed:
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'cLoss: {closs.val:>7.4f} ({closs.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})'
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name,
                            batch_idx,
                            last_idx,
                            batch_time=batch_time_m,
                            loss=losses_m,
                            closs=closses_m,
                            top1=top1_m,
                            top5=top5_m,
                            ))
                else:
                    _logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'cLoss: {closs.val:>7.4f} ({closs.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})'
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})\n'
                        'Fire_rate: {spike_rate}\n'
                        'Tot_spike: {tot_spike}\n'
                        'Thres: {threshold}\n'
                        'Mu: {mu_str}\n'
                        'Sigma: {sigma_str}\n'.format(
                            log_name,
                            batch_idx,
                            last_idx,
                            batch_time=batch_time_m,
                            loss=losses_m,
                            closs=closses_m,
                            top1=top1_m,
                            top5=top5_m,
                            spike_rate=spike_rate_avg_layer_str,
                            tot_spike=tot_spike,
                            threshold=threshold_str,
                            mu_str=mu_str,
                            sigma_str=sigma_str
                        ))

    # metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    if not args.distributed:
        if tsne:
            feature_vec = torch.cat(feature_vec)
            feature_cls = torch.cat(feature_cls)
            plot_tsne(feature_vec, feature_cls, os.path.join(args.output_dir, 't-sne-2d.eps'))
            plot_tsne_3d(feature_vec, feature_cls, os.path.join(args.output_dir, 't-sne-3d.eps'))
        if conf_mat:
            logits_vec = torch.cat(logits_vec)
            labels_vec = torch.cat(labels_vec)
            plot_confusion_matrix(logits_vec, labels_vec, os.path.join(args.output_dir, 'confusion_matrix.eps'))

    return metrics


if __name__ == '__main__':
    main()