import argparse
import time
import os
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
import logging
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.base.utils.criterions import *
from braincog.datasets.datasets import *

from mask_model import *
from utils import *

_logger = logging.getLogger('train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"),'c10'])
output_dir = get_outdir('./', 'train', exp_name)
setup_default_logging(log_path=os.path.join(output_dir, 'log.txt'))
_logger.info(exp_name)

config_parser = cfg = argparse.ArgumentParser(description='Training Config', add_help=False)
dataset='cifar10'
num_classes=10
step=8
encode='direct'
node_type='PLIFNode'
thresh=0.5
tau=2.0

torch.backends.cudnn.benchmark = True
devicee=0
seed=42
channels = 2
batch_size=50
epochs=300

lr=5e-3
linear_scaled_lr = lr * batch_size/ 1024.0
cfg.opt='adamw'
cfg.lr=linear_scaled_lr
cfg.weight_decay=0.01
cfg.momentum=0.9
cfg.epochs=epochs
cfg.sched='cosine'
cfg.min_lr=1e-5
cfg.warmup_lr=1e-6
cfg.warmup_epochs=5
cfg.cooldown_epochs=10
cfg.decay_rate=0.1

eval_metric='top1'
best_test = 0
best_testepoch = 0
best_testprun = 0
best_testepochprun = 0
epoch_prune = 1
rate_decay_epoch=30
NUM = 0

torch.cuda.set_device('cuda:%d' % devicee)
torch.manual_seed(seed)

model = my_cifar_model(step=step,encode_type=encode,node_type=node_type,num_classes=num_classes)
model = model.cuda()
print(model)

optimizer = create_optimizer(cfg, model)
lr_scheduler, num_epochs = create_scheduler(cfg, optimizer)

loader_train, loader_eval, mixup_active, mixup_fn = eval('get_%s_data' % dataset)(batch_size=batch_size, step=step)
train_loss_fn = UnilateralMse(1.)
validate_loss_fn = UnilateralMse(1.)


m = Mask(model,batch_size,step)
m.init_length()
trace=Trace(model,batch_size,step)

neuron_th,spines,bcm,epoch_trace = init(batch_size,convlayer,fclayer,size,fcsize)

def BCM_and_trace(NUM,trace,spikes,neuron_th,bcm,epoch_trace):
    NUM = NUM + 1
    csum,fcsum= trace.computing_trace(spikes)
    for i in range(1,len(convlayer)):
        index=convlayer[i]
        post1 = (csum[index] * (csum[index] - neuron_th[index]))
        hebb1 = torch.mm(post1.T, csum[index-1]) 
        bcm[index] = bcm[index] + hebb1
        neuron_th[index] = torch.div(neuron_th[index] * (NUM - 1) + csum[index], NUM)
        cs=torch.sum(csum[index],dim=0)
        epoch_trace[index] = epoch_trace[index] + cs

    for i in range(1,len(fclayer)):
        index = fclayer[i]
        post1 = (fcsum[index] * (fcsum[index] - neuron_th[index]))
        hebb1 = torch.mm(post1.T, fcsum[fclayer[i - 1]])
        bcm[index] = bcm[index] + hebb1
        neuron_th[index] = torch.div(neuron_th[index] * (NUM - 1) + fcsum[index], NUM)
        cs=torch.sum(fcsum[index],dim=0)
        epoch_trace[index] = epoch_trace[index] + cs
    return epoch_trace,bcm,NUM

def train_epoch(
        epoch, model, loader, optimizer, loss_fn,trace,NUM,bcm,neuron_th,epoch_trace,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
        output,spikes = model(inputs)

        epoch_trace,bcm,NUM = BCM_and_trace(NUM,trace,spikes,neuron_th,bcm,epoch_trace)

        loss = loss_fn(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses_m.update(loss.item(), inputs.size(0))
        top1_m.update(acc1.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx %100 == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            print("Train: epoch:",epoch,batch_idx,"/",len(loader),"loss:",losses_m.avg,"acc1:", top1_m.avg,"lr:",lr)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)]),epoch_trace,bcm,NUM

def validate(model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            inputs = inputs.type(torch.FloatTensor).cuda()
            target = target.cuda()

            output,spikes = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), output.size(0))
            if last_batch or batch_idx %100 == 0:
                print("Test: loss:",losses_m.avg,"acc1:", top1_m.avg)

            batch_time_m.update(time.time() - end)
            end = time.time()


    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


for epoch in range(epochs):

    train_metrics, epoch_trace, bcm, N = train_epoch(
        epoch, model, loader_train, optimizer, train_loss_fn,trace,NUM,bcm,neuron_th,epoch_trace,
        lr_scheduler=lr_scheduler)
    NUM = N

    for i in range(1,len(convlayer)):
        index=convlayer[i]
        bcmconv = torch.sum(bcm[index], dim=1)
        bcmconv=unit_tensor(bcmconv)
        traconv=unit_tensor(epoch_trace[index])
        spines[index]=bcmconv*traconv
    for i in range(1, len(fclayer)):
        index=fclayer[i]
        bcmfc = torch.sum(bcm[index], dim=1)
        bcmfc=unit_tensor(bcmfc)
        trafc=unit_tensor(epoch_trace[index])
        spines[index]=bcmfc*trafc

    if epoch>4:
        m.model = model
        m.init_mask(bcm,spines,epoch)
        m.do_mask()
        print("Done pruning")
        cc=m.if_zero()
        model = m.model

    eval_metrics = validate(model, loader_eval, validate_loss_fn)
    top1=eval_metrics['top1']
    if top1 > best_testprun:
        best_testprun = top1
        best_testepochprun =epoch
    if epoch%40==0:
        print('best acc:',best_testprun,'best epoch:',best_testepochprun)
    if epoch>4:
        _logger.info('*** epoch: {0} (pruning rate {1},acc:{2})'.format(epoch, cc,top1))
    if lr_scheduler is not None:
        lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
