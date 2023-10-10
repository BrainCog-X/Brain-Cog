import os
import sys
import time
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from braincog.model_zoo.NeuEvo.model_search import Network, calc_weight, calc_loss
from braincog.model_zoo.NeuEvo.architect import Architect
from separate_loss import ConvSeparateLoss, TriSeparateLoss, MseSeparateLoss
import utils

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy

from braincog.datasets.datasets import *
from braincog.base.utils.criterions import *

torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data/datasets',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or cifar 100 for searching')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--aux_loss_weight', type=float,
                    default=10.0, help='weight decay')
parser.add_argument('--device', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init-channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=6,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--single_level', action='store_true',
                    default=False, help='use single level')
parser.add_argument('--sep_loss', type=str, default='l2',
                    help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true',
                    default=False, help='use auto augmentation')
parser.add_argument('--parse_method', type=str,
                    default='bio_darts', help='parse the code method')
parser.add_argument('--op_threshold', type=float,
                    default=0.85, help='threshold for edges')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float,
                    default=1e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_lr_gamma', type=float, default=0.9,
                    help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

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

parser.add_argument('--img_size', default=32, type=int)
parser.add_argument('--smoothing', default=0.1, type=float)
parser.add_argument('--step', default=8, type=int)
parser.add_argument('--node-type', default='BiasPLIFNode', type=str)
parser.add_argument('--loss_fn', type=str, default='')
parser.add_argument('--back-connection', action='store_true')
parser.add_argument('--asbe', '--arch-search-begin-epoch',
                    type=int, default=0, dest='asbe')
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--spike-output',action='store_true')
parser.add_argument('--act-fun', type=str, default='GateGrad')
parser.add_argument('--suffix', default='', type=str)
args = parser.parse_args()

args.save = '/data/floyed/darts/logs/search/search-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"),
                                                                    args.suffix)
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    args.spike_output = False
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.device)
    # cudnn.benchmark = True
    torch.manual_seed(args.seed)
    # cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.device)
    logging.info("args = %s", args)
    run_start = time.time()
    start_epoch = 0
    dur_time = 0

    if args.loss_fn == 'mix':
        criterion_train = MixLoss(LabelSmoothingCrossEntropy(
            smoothing=args.smoothing).cuda())
        criterion_val = MixLoss(nn.CrossEntropyLoss())
    elif args.loss_fn == 'mse':
        criterion_train = UnilateralMse(1.)
        criterion_val = UnilateralMse(1.)
    else:
        criterion_train = LabelSmoothingCrossEntropy().cuda()
        criterion_val = nn.CrossEntropyLoss().cuda()

    criterion_train = ConvSeparateLoss(criterion_train, weight=args.aux_loss_weight) \
        if args.sep_loss == 'l2' else TriSeparateLoss(criterion_train, weight=args.aux_loss_weight)

    model = Network(args.init_channels, args.num_classes, args.layers, criterion_train,
                    steps=3, multiplier=3, stem_multiplier=3,
                    parse_method=args.parse_method, op_threshold=args.op_threshold,
                    step=args.step, node_type=args.node_type,
                    back_connection=args.back_connection, act_fun=args.act_fun,
                    dataset=args.dataset,
                    spike_output=False,
                    temporal_flatten=args.temporal_flatten)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        args.learning_rate,
        # momentum=args.momentum,
        weight_decay=args.weight_decay)

    # # train_transform, valid_transform = utils._data_transforms_cifar(args)
    # train_transform = build_transform(True, args.img_size)
    # valid_transform = build_transform(False, args.img_size)
    # train_data = dset.CIFAR10(
    #     root=args.data, train=True, download=True, transform=train_transform)
    #
    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))
    #
    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    #     pin_memory=True)
    #
    # valid_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(
    #         indices[split:num_train]),
    #     pin_memory=True)
    train_queue, valid_queue, _, _ = eval('get_%s_data' % args.dataset)(
        batch_size=args.batch_size,
        step=args.step,
        args=args,
        size=args.event_size,
        mix_up=args.mix_up,
        cut_mix=args.cut_mix,
        event_mix=args.event_mix,
        beta=args.cutmix_beta,
        prob=args.cutmix_prob,
        num=args.cutmix_num,
        noise=args.cutmix_noise,
        num_classes=args.num_classes,
        rand_aug=args.rand_aug,
        randaug_n=args.randaug_n,
        randaug_m=args.randaug_m,
        temporal_flatten=args.temporal_flatten
    )

    architect = Architect(model, args)

    # resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=model.alphas_normal.device)
            start_epoch = checkpoint['epoch']
            dur_time = checkpoint['dur_time']
            model_optimizer.load_state_dict(checkpoint['model_optimizer'])
            model.restore(checkpoint['network_states'])
            logging.info('=> loaded checkpoint \'{}\'(epoch {})'.format(
                args.resume, start_epoch))
        else:
            logging.info(
                '=> no checkpoint found at \'{}\''.format(args.resume))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, float(args.epochs), eta_min=args.learning_rate_min,
        last_epoch=-1 if start_epoch == 0 else start_epoch)
    if args.resume and os.path.isfile(args.resume):
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        logging.info(calc_weight(model.alphas_normal))
        logging.info(calc_loss(model.alphas_normal))
        model.update_history()

        # training and search the model
        train_acc, train_obj = train(epoch, train_queue, valid_queue, model, architect, criterion_train,
                                     model_optimizer)
        logging.info('train_acc %f', train_acc)

        # validation the model
        model.record_fire_rate = True
        model.reset_fire_rate_record()
        valid_acc, valid_obj = infer(valid_queue, model, criterion_val)
        fire_rate = model.get_fire_per_step()
        model.record_fire_rate = False
        logging.info('valid_fire_rate: {}'.format(fire_rate))
        logging.info('valid_acc %f', valid_acc)

        # save checkpoint
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'dur_time': dur_time + time.time() - run_start,
            'scheduler': scheduler.state_dict(),
            'model_optimizer': model_optimizer.state_dict(),
            'network_states': model.states(),
        }, is_best=False, save=args.save)
        logging.info('save checkpoint (epoch %d) in %s  dur_time: %s', epoch, args.save,
                     utils.calc_time(dur_time + time.time() - run_start))

        # save operation weights as fig
        utils.save_file(recoder=model.alphas_normal_history, path=os.path.join(args.save, 'normal'),
                        back_connection=args.back_connection)

    # save last operations
    np.save(os.path.join(os.path.join(args.save, 'normal_weight.npy')),
            calc_weight(model.alphas_normal).data.cpu().numpy())
    logging.info('save last weights done')


def train(epoch, train_queue, valid_queue, model, architect, criterion, model_optimizer):
    objs = utils.AvgrageMeter()
    objs1 = utils.AvgrageMeter()
    objs2 = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda(non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)
        # if epoch >= args.asbe:
            # Get a random minibatch from the search queue(validation set) with replacement
            # input_search, target_search = next(iter(valid_queue))
            # print(input.shape, target.shape)
            # print(input_search.shape, target_search.shape)
            # input_search = Variable(
            #     input_search, requires_grad=False).cuda(non_blocking=True)
            # target_search = Variable(
            #     target_search, requires_grad=False).cuda(non_blocking=True)
            # loss1, loss2 = architect.step(input_search, target_search)
        # else:
        loss1 = torch.tensor([0.])
        loss2 = torch.tensor([0.])

        model_optimizer.zero_grad()

        logits = model(input)
        aux_input = torch.cat(
            [calc_loss(model.alphas_normal)], dim=0)
        loss, _, _ = criterion(logits, target, aux_input)
        # loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        # Update the network parameters
        model_optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        objs1.update(loss1, n)
        objs2.update(loss2, n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d loss: %e top1: %f top5: %f',
                         step, objs.avg, top1.avg, top5.avg)
            logging.info('val cls_loss %e; spe_loss %e', objs1.avg, objs2.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda(non_blocking=True)
            target = Variable(target, volatile=True).cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


if __name__ == '__main__':
    main()
