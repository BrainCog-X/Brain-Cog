# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/4/28 14:56
# User      : Floyed
# Product   : PyCharm
# Project   : braincog
# File      : main_simplified.py
# explain   : Simplified training script. Remove support for DDP, IMAGENET, Augment, etc.

import argparse
import time

import timm.models
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
from braincog.utils import save_feature_map

import torch
import torch.nn as nn
import torchvision.utils
from torchvision import transforms
from timm.data import ImageDataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from maskcl2 import *
# from ptflops import get_model_complexity_info
from thop import profile, clever_format
from manipulate import permutate_image_pixels, SubDataset, TransformedDataset
torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')
from available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset
import copy
from vgg_snn import SNN,Taskmodel

# torch.cuda.set_device(9)

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='SNN Training and Evaluating')

# Model parameters
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--model', default='cifar_convnet', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--num-classes', type=int, default=100, metavar='N',
                    help='number of label classes (default: 10)')
parser.add_argument('--task_num', type=int, default=10, metavar='N',
                    help='number of label classes (default: 10)')

# Dataloader parameters
parser.add_argument('-b', '--batch-size', type=int, default=100, metavar='N',
                    help='inputs batch size for training (default: 128)')

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
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
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
parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
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

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--output', default='/home/hanbing/project/braincog/cls/bp_mnist2/', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')

# Spike parameters
parser.add_argument('--step', type=int, default=4, help='Simulation time step (default: 10)')
parser.add_argument('--encode', type=str, default='direct', help='Input encode method (default: direct)')
# neuron type
parser.add_argument('--node-type', type=str, default='LIFNode', help='Node type in network (default: PLIF)')
parser.add_argument('--act-fun', type=str, default='QGateGrad',
                    help='Surogate Function in node. Only for Surrogate nodes (default: AtanGrad)')
parser.add_argument('--thresh', type=float, default=.5, help='Firing threshold (default: 0.5)')
parser.add_argument('--tau', type=float, default=2., help='Attenuation coefficient (default: 2.)')

parser.add_argument('--loss-fn', type=str, default='ce', help='loss function (default: ce)')
parser.add_argument('--noisy-grad', type=float, default=0.,
                    help='Add noise to backward, sometime will make higher accuracy (default: 0.)')
parser.add_argument('--n_warm_up', type=int, default=0,
                    help='Warm up epoch, replace all node to ReLU to warm up weights in network before (default: 0)')
parser.add_argument('--spike-output', action='store_true', default=False,
                    help='Using mem output or spike output (default: False)')

# EventData Augmentation
parser.add_argument('--mix-up', action='store_true', help='Mix-up for event data (default: False)')
parser.add_argument('--cut-mix', action='store_true', help='CutMix for event data (default: False)')
parser.add_argument('--event-mix', action='store_true', help='EventMix for event data (default: False)')
parser.add_argument('--cutmix_beta', type=float, default=1.0, help='cutmix_beta (default: 1.)')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='cutmix_prib for event data (default: .5)')
parser.add_argument('--cutmix_num', type=int, default=1, help='cutmix_num for event data (default: 1)')
parser.add_argument('--cutmix_noise', type=float, default=0.,
                    help='Add Pepper noise after mix, sometimes work (default: 0.)')
parser.add_argument('--rand-aug', action='store_true',
                    help='Rand Augment for Event data (default: False)')
parser.add_argument('--randaug_n', type=int, default=3,
                    help='Rand Augment times n (default: 3)')
parser.add_argument('--randaug_m', type=int, default=15,
                    help='Rand Augment times n (default: 15) (0-30)')
parser.add_argument('--temporal-flatten', action='store_true',
                    help='Temporal flatten to channels. ONLY FOR EVENT DATA TRAINING BY ANN')
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

parser.add_argument('--suffix', type=str, default='',
                    help='Add an additional suffix to the save path (default: \'\')')


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

def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'MNIST' if name in ('MNIST28', 'MNIST32') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    # if permutation is not None:
    #     transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


def main():
    args, args_text = _parse_args()
    # args.no_spike_output = args.no_spike_output | args.cut_mix
    args.no_spike_output = True
    output_dir = ''
    output_base = args.output if args.output else './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        'SNN',
        'mnist',
        str(args.seed),
        'gwu',
        # str(args.img_size)
    ])
    output_dir = get_outdir(output_base, 'train', exp_name)
    args.output_dir = output_dir
    setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))

    torch.cuda.set_device('cuda:%d' % args.device)

    torch.manual_seed(args.seed)

    model = SNN(
        num_classes=args.num_classes,
        dataset=args.dataset,
        step=args.step,
        encode_type=args.encode,
        node_type=eval(args.node_type),
        threshold=args.thresh,
        tau=args.tau,
        spike_output=not args.no_spike_output,
        act_fun=args.act_fun,
        temporal_flatten=args.temporal_flatten,
        layer_by_layer=args.layer_by_layer,
        batch_size=args.batch_size,
        task_num=args.task_num
    )


    print(model)
    # for n,p in enumerate(model.parameters()):
    #     print(n,p.size())
    if 'dvs' in args.dataset:
        args.channels = 2
    elif 'mnist' in args.dataset:
        args.channels = 1
    else:
        args.channels = 3
    # flops, params = profile(model, inputs=(torch.randn(1, args.channels, args.img_size, args.img_size),), verbose=False)
    # _logger.info('flops = %fM', flops / 1e6)
    # _logger.info('param size = %fM', params / 1e6)

    linear_scaled_lr = args.lr * args.batch_size / 1024.0
    args.lr = linear_scaled_lr

    model = model.cuda()
    taskmodel=Taskmodel(num_classes=args.num_classes,task_num=args.task_num,batch_size=args.batch_size)
    taskmodel = taskmodel.cuda()
    optimizer = create_optimizer(args, model)
    optimizer_re =torch.optim.Adam(taskmodel.parameters())

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        # checkpoint = torch.load(args.resume, map_location='cpu')
        # model.load_state_dict(checkpoint['state_dict'], False)
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer)

    if args.node_resume:
        ckpt = torch.load(args.node_resume, map_location='cpu')
        model.load_node_weight(ckpt, args.node_trainable)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    m = Mask(model)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    _logger.info('Scheduled epochs: {}'.format(num_epochs))
    batch_size=args.batch_size
    data_dir = '/home/hanbing/project/BP-for-SpikingNN-master3/MNIST/' 


    trainset = get_dataset('MNIST', type="train", dir=data_dir)
    testset = get_dataset('MNIST', type="test", dir=data_dir)
    permutations = [np.random.permutation(784) for _ in range(args.task_num)]
    #permutations.append(np.array([i for i in range(784)]))
    train_datasets = []
    test_datasets = []
    for context_id, perm in enumerate(permutations):
        target_transform = None
        train_datasets.append(TransformedDataset(
            trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
            target_transform=target_transform
        ))
        test_datasets.append(TransformedDataset(
            testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
            target_transform=target_transform
        ))

    train_data = []
    test_data = []
    train_rdata = []
    for task in range(len(train_datasets)):
        train_data.append(DataLoader(train_datasets[task], batch_size=batch_size, shuffle=True, drop_last=True, **({'num_workers': 0, 'pin_memory': True})))
        test_data.append(DataLoader(test_datasets[task], batch_size=batch_size, shuffle=True, drop_last=True, **({'num_workers': 0, 'pin_memory': True})))
        train_rdata.append(DataLoader(train_datasets[task], batch_size=batch_size, shuffle=False, drop_last=True,**({'num_workers': 0, 'pin_memory': True})))
    # def yyy(y,task):
    #     y=task
    #     return y

    # retrain_data=[]
    # for i in range(10):
    #     if i==0:
    #         retrain_data[i]=train_datasets[i]
    #     else:
    #         transform=transforms.Lambda(lambda y:yyy(y,i))
    #         retrain_data[i]=torch.utils.data.ConcatDataset([retrain_data[i-1], train_datasets[i]])
    print(trainset)
    print(len(train_datasets[0]))

    if args.loss_fn == 'mse':
        train_loss_fn = UnilateralMse(1.)
        validate_loss_fn = UnilateralMse(1.)

    else:

        train_loss_fn = nn.CrossEntropyLoss().cuda()

        validate_loss_fn = nn.CrossEntropyLoss().cuda()

    if args.loss_fn == 'mix':
        train_loss_fn = MixLoss(train_loss_fn)
        validate_loss_fn = MixLoss(validate_loss_fn)

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None

    saver = CheckpointSaver(
        model=model, optimizer=optimizer, args=args,
        checkpoint_dir=output_dir, recovery_dir=output_dir)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    loader_his=[]
    task_ready={}
    for index, item in enumerate(model.parameters()):
        if len(item.size()) > 1 and index<=10:
            task_ready[index]=torch.zeros(item.size(),device=device)
    try:  # train the model
        task_count=0
        regularization_terms= {}
        for task in range(len(train_datasets)):
            print("Task:",task)
            if task==0:
                m.model = model
                mat=m.init_length()
                model = m.model
                epochs=50
            else:
                m.model = model
                mat,task_ready,taskmaskk,taskww=m.init_grow(task)
                model = m.model
                epochs=30
            ta_his=[i for i in range(task+1)]

            for epoch in range(epochs):
                loader_train = iter(train_data[task])
                if task==0:
                    train_epoch(epoch, task, model, loader_train, optimizer, train_loss_fn, args,mat,task_ready,taskww=None,
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,regularization_terms=regularization_terms)
                else:
                    train_epoch(epoch, task, model, loader_train, optimizer, train_loss_fn, args,mat,task_ready,taskww=taskww,
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,regularization_terms=regularization_terms)

                print(epoch)

                if epoch>0:
                    m.model = model
                    m.init_mask(task,epoch)
                    mat=m.do_mask(task)
                    model = m.model

                for t in ta_his:
                    loader_his=iter(test_data[t])
                    validate(t, model, loader_his, validate_loss_fn, args,mat)

                for t in ta_his:
                    reloader_his=iter(train_rdata[task])
                    retrain_epoch(epoch, t,task, model, taskmodel,reloader_his, optimizer_re, train_loss_fn, args,mat,task_ready,taskww=None,
                        lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,regularization_terms=regularization_terms)

                for t in ta_his:
                    loader_his=iter(test_data[t])
                    revalidate(t, model, taskmodel,loader_his, validate_loss_fn, args,mat)

                cc=m.if_zero()
                _logger.info('*** epoch: {0}, task: {1}, pruning: {2}'.format(epoch,task, cc))
            p_index=m.record()

                    
            # task_param = {}
            # for n, p in enumerate(model.parameters()):
            #     if len(p.size())>=1:
            #         task_param[n] = p.clone().detach()
            # loader_train = iter(DataLoader(train_datasets[task], batch_size=batch_size, shuffle=True, drop_last=True, **({'num_workers': 0, 'pin_memory': True})))
            # importance = calculate_importance(model,loader_train,task,train_loss_fn)
            # task_count += 1
            # regularization_terms[task_count] = {'importance':importance, 'task_param':task_param}

    except KeyboardInterrupt:
        pass
    # if best_metric is not None:
    #     _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(
        epoch, task,model, loader, optimizer, loss_fn, args,mat,task_ready,taskww=None,
        lr_scheduler=None, saver=None, output_dir='',regularization_terms={}):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()

        output = model(inputs, mat)
        t_preds = output[task].cuda()
        loss = loss_fn(t_preds, target)

        acc1, acc5 = accuracy(t_preds, target, topk=(1, 5))

        losses_m.update(loss.item(), inputs.size(0))
        top1_m.update(acc1.item(), inputs.size(0))
        top5_m.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        # for index, item in enumerate(model.parameters()):
        #     if len(item.size()) > 1 and index<=40:
        #         gradmask=torch.where(task_ready[index]>0,0.0,1.0)
        #         item.grad=item.grad*gradmask
        optimizer.step()
        for index, item in enumerate(model.parameters()):
            if len(item.size()) > 1 and index<=10:
                if index<10:
                    ready=task_ready[index].view(task_ready[index].size()[0],-1)
                    ready=torch.sum(ready,dim=1)
                else:
                    ready=torch.sum(task_ready[index],dim=1)
                windex=torch.nonzero(ready>0)
                for i in range(len(windex)):
                    item.data[windex[i]]=taskww[index][windex[i]]

        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % 100 == 0:
            # lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            # lr = sum(lrl) / len(lrl)

            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  ' 
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    top1=top1_m, top5=top5_m,
                    batch_time=batch_time_m,
                    rate=inputs.size(0) / batch_time_m.val,
                    rate_avg=inputs.size(0)  / batch_time_m.avg,
                    data_time=data_time_m))

        # if saver is not None and args.recovery_interval and (
        #         last_batch or (batch_idx + 1) % args.recovery_interval == 0):
        #     saver.save_recovery(epoch, batch_idx=batch_idx)

    #     if lr_scheduler is not None:
    #         lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

    #     end = time.time()
    # # end for

    # if hasattr(optimizer, 'sync_lookahead'):
    #     optimizer.sync_lookahead()

    # return OrderedDict([('loss', losses_m.avg)])


def validate(task, model, loader, loss_fn, args, mat,log_suffix='', visualize=False, spike_rate=False):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        last_idx = len(loader) - 1
        for batch_idx, (inputs, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            inputs = inputs.type(torch.FloatTensor).cuda()
            target = target.cuda()

            output = model(inputs,mat)
            t_preds = output[task]
            loss = loss_fn(t_preds, target)
            acc1, acc5 = accuracy(t_preds, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

        log_name = 'Test'+str(task) + log_suffix
        _logger.info(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  ' 
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
            'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m,
                loss=losses_m, top1=top1_m, top5=top5_m))

    # metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    # return metrics

def retrain_epoch(
        epoch, task,task2,model, taskmodel,loader, optimizer, loss_fn, args,mat,task_ready,taskww=None,
        lr_scheduler=None, saver=None, output_dir='',regularization_terms={}):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    taskmodel.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    if task==task2:
        bstop=20
        log=20
    else:
        bstop=20
        log=20
    for batch_idx, (inputs, target) in enumerate(loader):
        if batch_idx>bstop:
            break
        last_batch = batch_idx == bstop
        data_time_m.update(time.time() - end)
        inputs = inputs.type(torch.FloatTensor).cuda()
        targets=task*torch.ones(args.batch_size,dtype=torch.int64)
        target = torch.zeros(args.batch_size, int(args.num_classes/args.task_num)).scatter_(1, targets.view(-1, 1), 1).cuda()

        output= model(inputs, mat)
        tout=taskmodel(output.detach())
        t_preds = tout.cuda()
        loss = loss_fn(t_preds, target)

        acc1, acc5 = accuracy(t_preds, targets.cuda(), topk=(1, 5))

        losses_m.update(loss.item(), inputs.size(0))
        top1_m.update(acc1.item(), inputs.size(0))
        top5_m.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or (batch_idx+1) %log == 0:

            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  ' 
                'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    top1=top1_m, top5=top5_m,
                    batch_time=batch_time_m,
                    rate=inputs.size(0) / batch_time_m.val,
                    rate_avg=inputs.size(0)  / batch_time_m.avg,
                    data_time=data_time_m))

def revalidate(task, model,taskmodel, loader, loss_fn, args, mat,log_suffix='', visualize=False, spike_rate=False):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    model.eval()
    taskmodel.eval()
    end = time.time()
    with torch.no_grad():
        last_idx = len(loader) - 1
        for batch_idx, (inputs, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            inputs = inputs.type(torch.FloatTensor).cuda()
            target = target.cuda()

            output = model(inputs,mat)
            tout=taskmodel(output)
            t=torch.max(tout,dim=1)
            t_preds=torch.zeros(args.batch_size, int(args.num_classes/args.task_num))
            for i in range(args.batch_size):
                t_preds[i]=output[t[1][i],i]
            # t_preds = output[t[1]]
            loss = loss_fn(t_preds.cuda(), target)
            acc1, acc5 = accuracy(t_preds.cuda(), target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

        log_name = 'Test'+str(task) + log_suffix
        _logger.info(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  ' 
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
            'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m,
                loss=losses_m, top1=top1_m, top5=top5_m))

if __name__ == '__main__':
    main()
