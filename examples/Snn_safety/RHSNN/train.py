import argparse
import copy
import logging
import os
import sys
import time
from evaluate import evaluate_attack
import torchattacks
from my_node import RHLIFNode, RHLIFNode2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sew_resnet import SEWResNet19, BasicBlock, Bottleneck
from braincog.base.node.node import *

from utils import (evaluate_standard, cifar10_std, cifar10_mean,
                   orthogonal_retraction)

from utils import (clamp, get_norm_stat,
                   get_loaders)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='/mnt/data/datasets', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--network', default='ResNet18', type=str)
    parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--lr_schedule', default='cosine', choices=['cyclic', 'multistep', 'cosine'])
    parser.add_argument('--lr_min', default=0., type=float)
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=4, type=int)
    parser.add_argument('--alpha', default=4, type=float, help='Step size')
    parser.add_argument('--save_dir', default='ckpt', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    parser.add_argument('--attack_iters', default=1, type=int, help='Attack iterations')

    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')

    parser.add_argument('--beta', default=0.004, type=float)
    parser.add_argument('--adv_training', action='store_true',
                        help='if adv training')

    parser.add_argument('--time_step', default=8, type=int)
    parser.add_argument('--SR', action='store_true')
    parser.add_argument('--node_type', default='LIF', type=str)
    parser.add_argument('--parseval', action='store_true', help='if use different norm for different layers')

    return parser.parse_args()


def main():
    args = get_args()
    device = args.device
    torch.cuda.set_device(device)
    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    mu, std, upper_limit, lower_limit = get_norm_stat(cifar10_mean, cifar10_std)

    path = os.path.join('./ckpt', args.dataset, args.network)
    args.save_dir = os.path.join(path, args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # get data loader
    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker)
    train_loader_e, test_loader_e, dataset_normalization_e = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker, norm=False)

    # adv training attack setting
    epsilon = ((args.epsilon / 255.) / std).to(device)
    alpha = ((args.alpha / 255.) / std).to(device)

    node = LIFNode
    if args.node_type == 'RHLIF':
        node = RHLIFNode
    elif args.node_type == 'RHLIF2':
        node = RHLIFNode2
    # setup network

    model = SEWResNet19(BasicBlock, [3, 3, 2], cnf='ADD', node_type=node, step=args.time_step, num_classes=args.num_classes,
                            layer_by_layer=True, act_fun=AtanGrad)
    model.to(device)

    # model = torch.nn.DataParallel(model)
    # logger.info(model)

    # setup optimizer, loss function, LR scheduler
    # opt = torch.optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    if args.parseval:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=0)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)


    criterion = nn.CrossEntropyLoss()

    if args.lr_schedule == 'cyclic':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        lr_steps = args.epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_pgd_acc = 0
    best_clean_acc = 0
    test_acc_best_pgd = 0

    start_epoch = 0

    # Start training
    start_train_time = time.time()

    for epoch in range(start_epoch, args.epochs):

        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.train()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):

            _iters = epoch * len(train_loader) + i

            X, y = X.to(device), y.to(device)
            if args.adv_training:
                # init delta
                delta = torch.zeros_like(X).to(device)
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_((-epsilon[j][0][0] / 10).item(), (epsilon[j][0][0]/10).item())
                delta.data = clamp(delta, lower_limit.to(device) - X, upper_limit.to(device) - X)
                delta.requires_grad = True

                # pgd attack
                for _ in range(args.attack_iters):
                    output = model(X + delta)
                    # model.random_reset_step = 0
                    loss = criterion(output, y)

                    loss.backward()

                    grad = delta.grad.detach()

                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit.to(device) - X, upper_limit.to(device) - X)
                    delta.grad.zero_()

                delta = delta.detach()
                X_adv = X + delta[:X.size(0)]
            else:
                X_adv = X

            if args.SR:
                X_adv.requires_grad_(True)

                outputs = model(X_adv)
                out = outputs.gather(1, y.unsqueeze(1)).squeeze()  # choose
                batch = []
                inds = []
                for j in range(len(outputs)):
                    mm, ind = torch.cat([outputs[j, :y[j]], outputs[j, y[j] + 1:]], dim=0).max(0)
                    f = torch.exp(out[j]) / (torch.exp(out[j]) + torch.exp(mm))
                    batch.append(f)
                    inds.append(ind.item())
                f1 = torch.stack(batch, dim=0)

                loss1 = criterion(outputs, y)

                dx = torch.autograd.grad(f1, X_adv, grad_outputs=torch.ones_like(f1, device=device), retain_graph=True)[0]
                X_adv.requires_grad_(False)

                v = dx.detach().sign()

                x2 = X_adv + 0.01 * v

                outputs2 = model(x2)

                out = outputs2.gather(1, y.unsqueeze(1)).squeeze()  # choose
                batch = []
                for j in range(len(outputs2)):
                    mm = torch.cat([outputs2[j, :y[j]], outputs2[j, y[j] + 1:]], dim=0)[inds[j]]
                    f = torch.exp(out[j]) / (torch.exp(out[j]) + torch.exp(mm))
                    batch.append(f)
                f2 = torch.stack(batch, dim=0)

                dl = (f2 - f1) / 0.01
                loss2 = dl.pow(2).mean()
                loss = loss1 + 0.001 * loss2
                loss = loss.mean()
            else:
                output = model(X_adv)
                loss = criterion(output, y)

            opt.zero_grad()
            loss.backward()

            opt.step()
            if args.parseval:
                orthogonal_retraction(model, args.beta)
            # for name, param in model.named_parameters():
            #     if 'sigma' in name:  # 查找包含 'sigma' 的参数
            #         param.data = torch.clamp(param.data, 0.0, 1.5)  # 约束参数范围
            #         if i==0:
            #             print(param)
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            if i % 50 == 0:
                logger.info("Iter: [{:d}][{:d}/{:d}]\t"
                            "Loss {:.3f} ({:.3f})\t"
                            "Prec@1 {:.3f} ({:.3f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    loss.item(),
                    train_loss / train_n,
                    (output.max(1)[1] == y).sum().item() / y.size(0),
                    train_acc / train_n)
                )

        scheduler.step()

        logger.info('Evaluating with standard images...')
        test_loss, test_acc = evaluate_standard(test_loader, model, args)
        logger.info(
            'Test Loss: %.4f  \t Test Acc: %.4f',
            test_loss, test_acc)
        if test_acc > best_clean_acc:
            best_clean_acc = (
                test_acc)

            torch.save(model.state_dict(), os.path.join(args.save_dir, 'weight_c.pth'))

        # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 5, 1, args)
        if epoch > args.epochs - 10:
            logger.info('Evaluating with APGD Attack...')
            model.normalize = dataset_normalization_e
            atk = torchattacks.APGD(model, norm='Linf', eps=8 / 255, steps=10)
            pgd_loss, pgd_acc = evaluate_attack(model, test_loader_e, args, atk, 'APGD', logger)
            model.normalize = dataset_normalization

            if pgd_acc > best_pgd_acc:
                best_pgd_acc = pgd_acc
                test_acc_best_pgd = test_acc

                torch.save(model.state_dict(), os.path.join(args.save_dir, 'weight_r.pth'))
            logger.info(
                    'PGD Loss: %.4f \t PGD Acc: %.4f \n Best PGD Acc: %.4f \t Test Acc of best PGD ckpt: %.4f',
                    pgd_loss, pgd_acc, best_pgd_acc, test_acc_best_pgd)


    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)


if __name__ == "__main__":
    main()
