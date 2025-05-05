import argparse
import copy
import logging
import os
import sys
import time
from my_node import RHLIFNode, RHLIFNode2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sew_resnet import SEWResNet19, BasicBlock
from braincog.base.node.node import *

from utils import evaluate_standard

from utils import get_loaders

import torchattacks

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--data_dir', default='/mnt/data/datasets', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--network', default='ResNet18', type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save log')
    parser.add_argument('--attack_type', default='pgd')
    parser.add_argument('--time_step', default=8, type=int)
    parser.add_argument('--node_type', default='LIF', type=str)
    return parser.parse_args()

def evaluate_attack(model, test_loader, args, atk, atk_name, logger):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    device = args.device

    test_loader = iter(test_loader)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format, ncols=80)
    for i in pbar:
        X, y = next(test_loader)
        X, y = X.to(device), y.to(device)
        X_adv = atk(X, y)  # advtorch
        with torch.no_grad():
            output = model(X_adv)
        loss = F.cross_entropy(output, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    pgd_acc = test_acc / n
    pgd_loss = test_loss / n

    logger.info(atk_name)
    logger.info('adv: %.4f \t', pgd_acc)

    return pgd_loss, pgd_acc

def main():
    args = get_args()

    args.save_dir = os.path.join('logs', args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)


    log_path = os.path.join(args.save_dir, 'output_test.log')

    handlers = [logging.FileHandler(log_path, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)

    logger.info(args)

    # assert type(args.pretrain) == str and os.path.exists(args.pretrain)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    logger.info('Dataset: %s', args.dataset)

    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker, norm=False)
    node = LIFNode
    if args.node_type == 'RHLIF':
        node = RHLIFNode
    elif args.node_type == 'RHLIF2':
        node = RHLIFNode2

    # setup network
    model = SEWResNet19(BasicBlock, [3, 3, 2], cnf='ADD', node_type=node, step=args.time_step, num_classes=args.num_classes,
                        layer_by_layer=True, act_fun=AtanGrad, data_norm=dataset_normalization)
    # print(model)

    # load pretrained model
    path = os.path.join('./ckpt', args.dataset, args.network)
    args.pretrain = os.path.join(path, args.pretrain)
    pretrained_model = torch.load(args.pretrain, map_location=args.device, weights_only=False)
    model.load_state_dict(pretrained_model, strict=False)
    model.to(args.device)
    model.eval()
    # for name, param in model.named_parameters():
    #     if 'sigma' in name:  # 查找包含 'sigma' 的参数
    #         param.data = torch.as_tensor(0.0, device=args.device)
    #     if 'alpha' in name:  # 查找包含 'sigma' 的参数
    #         param.data = torch.as_tensor(2.0, device=args.device)

    logger.info('Evaluating with standard images...')
    _, nature_acc = evaluate_standard(test_loader, model, args)
    logger.info('Nature Acc: %.4f \t', nature_acc)

    if args.attack_type == 'eotpgd':
        atk = torchattacks.EOTPGD(model, eps=8 / 255, alpha=(16/50) / 255, steps=50, random_start=True, eot_iter=10)
        evaluate_attack(model, test_loader, args, atk, 'eotpgd', logger)
    elif args.attack_type[0:3] == 'pgd':
        steps = int(''.join(filter(str.isdigit, args.attack_type)))
        atk = torchattacks.PGD(model, eps=8 / 255, alpha=(16/steps) / 255, steps=steps, random_start=True)
        evaluate_attack(model, test_loader, args, atk, args.attack_type, logger)
    elif args.attack_type == 'apgd':
        atk = torchattacks.APGD(model, eps=8 / 255, steps=50, eot_iter=10)
        evaluate_attack(model, test_loader, args, atk, 'apgd', logger)
    elif args.attack_type == 'fgsm':
        atk = torchattacks.FGSM(model, eps=8/255)
        evaluate_attack(model, test_loader, args, atk, 'fgsm', logger)
    elif args.attack_type == 'mifgsm':
        atk = torchattacks.MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=5, decay=1.0)
        evaluate_attack(model, test_loader, args, atk, 'mifgsm', logger)
    elif args.attack_type == 'autoattack':
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=args.num_classes)
        evaluate_attack(model, test_loader, args, atk, 'autoattack', logger)
    elif args.attack_type == 'all':
        atk = torchattacks.FGSM(model, eps=8 / 255)
        evaluate_attack(model, test_loader, args, atk, 'fgsm', logger)
        atk = torchattacks.APGD(model, eps=8 / 255, steps=10)
        evaluate_attack(model, test_loader, args, atk, 'apgd', logger)
        atk = torchattacks.PGD(model, eps=8 / 255, alpha=1.6 / 255, steps=10, random_start=True)
        evaluate_attack(model, test_loader, args, atk, 'pgd', logger)
        atk = torchattacks.MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=5, decay=1.0)
        evaluate_attack(model, test_loader, args, atk, 'mifgsm', logger)
        atk = torchattacks.AutoAttack(model, norm='Linf', eps=8 / 255, version='standard',
                                      n_classes=args.num_classes)
        evaluate_attack(model, test_loader, args, atk, 'autoattack', logger)
    elif args.attack_type == 'step_test':
        for steps in [10,30,50,70,90,110]:
            atk = torchattacks.PGD(model, eps=8 / 255, alpha=(16 / steps) / 255, steps=steps, random_start=True)
            pgd_loss, pgd_acc = evaluate_attack(model, test_loader, args, atk, f'pgd{steps}', logger)
            atk = torchattacks.APGD(model, eps=8 / 255, steps=steps)
            apgd_loss, apgd_acc = evaluate_attack(model, test_loader, args, atk, f'apgd{steps}', logger)

    elif args.attack_type == 'eot_test':

        for steps in [1,10,20,30]:
            atk = torchattacks.EOTPGD(model, eps=8 / 255, alpha=(16 / 10) / 255, steps=10, random_start=True, eot_iter=steps)
            pgd_loss, pgd_acc = evaluate_attack(model, test_loader, args, atk, f'eot{steps}_pgd', logger)
            atk = torchattacks.APGD(model, eps=8 / 255, steps=10, eot_iter=steps)
            apgd_loss, apgd_acc = evaluate_attack(model, test_loader, args, atk, f'eot{steps}_apgd', logger)

    elif args.attack_type == 'intensity_test':

        for intensity in [2, 4, 6, 8, 10, 12, 14, 16]:
            atk = torchattacks.APGD(model, eps=intensity / 255, steps=10)
            pgd_loss, pgd_acc = evaluate_attack(model, test_loader, args, atk, f'{intensity}_apgd', logger)


    logger.info('Testing done.')


if __name__ == "__main__":
    main()
