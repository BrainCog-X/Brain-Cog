import os
import sys
import time
import logging
import torch
import utils as dutils
import argparse
import numpy as np
import torch.utils
import torch.nn as nn
from braincog.model_zoo.darts import genotypes
from braincog.model_zoo.darts.model import NetworkCIFAR as Network
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from thop import profile

from braincog.datasets.datasets import build_transform
from braincog.base.utils.criterions import UnilateralMse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/data/datasets',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or cifar 100 for training')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--device', type=int, default=0, help='gpu device id')
parser.add_argument('--multi-gpus', action='store_true',
                    default=False, help='use multi gpus')
parser.add_argument('--parse_method', type=str,
                    default='darts', help='experiment name')
parser.add_argument('--epochs', type=int, default=600,
                    help='num of training epochs')
parser.add_argument('--init-channels', type=int,
                    default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=16,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true',
                    default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float,
                    default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true',
                    default=False, help='use auto augmentation')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS',
                    help='which architecture to use')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--img_size', default=32, type=int)
parser.add_argument('--step', default=8, type=int)
parser.add_argument('--node-type', default='PLIFNode', type=str)
parser.add_argument('--suffix', default='', type=str)


class TrainNetwork(object):
    """The main train network"""

    def __init__(self, args):
        super(TrainNetwork, self).__init__()
        self.args = args
        self.dur_time = 0
        self._init_log()
        self._init_device()
        self._init_data_queue()
        self._init_model()

    def _init_log(self):
        self.args.save = '/data/floyed/darts/logs/eval/' + self.args.arch + '/' + 'cifar10' + '/eval-{}-{}-{}'.format(
            self.args.save, time.strftime('%Y%m%d-%H%M'), args.suffix)
        dutils.create_exp_dir(self.args.save, scripts_to_save=None)

        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        self.logger = logging.getLogger('Architecture Training')
        self.logger.addHandler(fh)

    def _init_device(self):
        if not torch.cuda.is_available():
            self.logger.info('no gpu device available')
            sys.exit(1)
        np.random.seed(self.args.seed)
        self.device_id = self.args.device
        self.device = torch.device('cuda:{}'.format(
            0 if self.args.multi_gpus else self.device_id))
        cudnn.benchmark = True
        torch.manual_seed(self.args.seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(self.args.seed)
        logging.info('gpu device = %d' % self.args.device)
        logging.info("args = %s", self.args)

    def _init_data_queue(self):
        train_transform = build_transform(True, args.img_size)
        valid_transform = build_transform(False, args.img_size)
        if self.args.dataset == 'cifar10':
            train_data = dset.CIFAR10(
                root=self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR10(
                root=self.args.data, train=False, download=True, transform=valid_transform)
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            train_data = dset.CIFAR100(
                root=self.args.data, train=True, download=True, transform=train_transform)
            valid_data = dset.CIFAR100(
                root=self.args.data, train=False, download=True, transform=valid_transform)
            self.num_classes = 100
        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

        self.valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    def _init_model(self):
        genotype = eval('genotypes.%s' % self.args.arch)
        model = Network(self.args.init_channels,
                        self.num_classes,
                        self.args.layers,
                        self.args.auxiliary,
                        genotype,
                        self.args.parse_method,
                        step=args.step,
                        node_type=args.node_type
                        )
        flops, params = profile(model, inputs=(
            torch.randn(1, 3, 32, 32),), verbose=False)
        self.logger.info('flops = %fM', flops / 1e6)
        self.logger.info('param size = %fM', params / 1e6)

        # Try move model to multi gpus
        if torch.cuda.device_count() > 1 and self.args.multi_gpus:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)
        self.model = model.to(self.device)

        # criterion = nn.CrossEntropyLoss()
        criterion = UnilateralMse(1.)
        self.criterion = criterion.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        self.best_acc_top1 = 0
        # optionally resume from a checkpoint
        if self.args.resume:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint {}".format(self.args.resume))
                checkpoint = torch.load(
                    self.args.resume, map_location=self.device)
                self.dur_time = checkpoint['dur_time']
                self.args.start_epoch = checkpoint['epoch']
                self.best_acc_top1 = checkpoint['best_acc_top1']
                self.args.drop_path_prob = checkpoint['drop_path_prob']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(self.args.epochs), eta_min=0,
                                                                    last_epoch=-1 if self.args.start_epoch == 0 else self.args.start_epoch)
        # reload the scheduler if possible
        if self.args.resume and os.path.isfile(self.args.resume):
            checkpoint = torch.load(self.args.resume)
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def run(self):
        self.logger.info('args = %s', self.args)
        run_start = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.scheduler.step()
            self.logger.info('epoch % d / %d  lr %e', epoch,
                             self.args.epochs, self.scheduler.get_lr()[0])

            self.model.drop_path_prob = self.args.drop_path_prob * epoch / self.args.epochs

            train_acc, train_obj = self.train()
            self.logger.info('train loss %e, train acc %f',
                             train_obj, train_acc)

            valid_acc_top1, valid_acc_top5, valid_obj = self.infer()
            self.logger.info('valid loss %e, top1 valid acc %f top5 valid acc %f',
                             valid_obj, valid_acc_top1, valid_acc_top5)
            self.logger.info('best valid acc %f', self.best_acc_top1)

            is_best = False
            if valid_acc_top1 > self.best_acc_top1:
                self.best_acc_top1 = valid_acc_top1
                is_best = True

            dutils.save_checkpoint({
                'epoch': epoch + 1,
                'dur_time': self.dur_time + time.time() - run_start,
                'state_dict': self.model.state_dict(),
                'drop_path_prob': self.args.drop_path_prob,
                'best_acc_top1': self.best_acc_top1,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()
            }, is_best, self.args.save)
        self.logger.info('train epoches %d, best_acc_top1 %f, dur_time %s',
                         self.args.epochs, self.best_acc_top1,
                         dutils.calc_time(self.dur_time + time.time() - run_start))

    def train(self):
        objs = dutils.AvgrageMeter()
        top1 = dutils.AvgrageMeter()
        top5 = dutils.AvgrageMeter()

        self.model.train()

        for step, (input, target) in enumerate(self.train_queue):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            self.optimizer.zero_grad()
            logits, logits_aux = self.model(input)
            loss = self.criterion(logits, target)
            if self.args.auxiliary:
                loss_aux = self.criterion(logits_aux, target)
                loss += self.args.auxiliary_weight * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            prec1, prec5 = dutils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0:
                self.logger.info('train %03d %e %f %f', step,
                                 objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg

    def infer(self):
        objs = dutils.AvgrageMeter()
        top1 = dutils.AvgrageMeter()
        top5 = dutils.AvgrageMeter()
        self.model.eval()
        with torch.no_grad():
            for step, (input, target) in enumerate(self.valid_queue):
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                logits, _ = self.model(input)
                loss = self.criterion(logits, target)

                prec1, prec5 = dutils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                if step % self.args.report_freq == 0:
                    self.logger.info('valid %03d %e %f %f',
                                     step, objs.avg, top1.avg, top5.avg)
            return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    args = parser.parse_args()
    train_network = TrainNetwork(args)
    train_network.run()
