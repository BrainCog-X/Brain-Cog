import sys
import numpy as np
import argparse
import time
import timm.models
import yaml
import os
import logging
from random import choice
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from micro_encoding import ops
from braincog.base.node.node import *
from braincog.utils import *
from braincog.base.utils.criterions import *
from braincog.datasets.datasets import *
from braincog.model_zoo.resnet import *
from braincog.model_zoo.convnet import *
from braincog.utils import save_feature_map, setup_seed
from braincog.base.utils.visualization import plot_tsne_3d, plot_tsne, plot_confusion_matrix
import micro_encoding
from pymop.problem import Problem
import torch
from thop import profile
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from pymoo.optimize import minimize
from timm.data import create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import load_checkpoint, create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from torchprofile import profile_macs
import copy
import torch.backends.cudnn as cudnn
import warnings
warnings.simplefilter("ignore")

def train_motifs(args,gen,arch_dir,genome,_logger,args_text,devices,ms,glob):
    if args.bns:
        from cellmodel import NetworkCIFAR_
    else:
        from cell123model import NetworkCIFAR_
    
    # qw=np.where(args.glob_con[0]==1)
    # ccc=np.array([1,0,0,0])
    # for i in qw:
    #     ccc[i[0]]=1
    #     ddd=np.where(args.glob_con[i[0]]==1)
    #     if len(ddd[0])!=0:
    #         for j in ddd:
    #             ccc[j[0]]=1
    #             www=np.where(args.glob_con[j[0]]==1)
    #             if len(www[0])!=0:
    #                 for k in www:
    #                     ccc[k[0]]=1


    test_motifs,ids = micro_encoding.decode_motif(args.layers*ms,args.bits,genome.astype(int))

    

    


        


    # if gen==-1:
    args.epochs=args.eval_epochs
    # else:
    #     args.epochs=args.eval_epochs

    try:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            dataset=args.dataset,
            step=args.step,
            encode_type=args.encode,
            node_type=eval(args.node_type),
            threshold=args.threshold,
            tau=args.tau,
            sigmoid_thres=args.sigmoid_thres,
            requires_thres_grad=args.requires_thres_grad,
            spike_output=not args.no_spike_output,
            C=args.init_channels,
            layers=args.layers*ms,
            auxiliary=args.auxiliary,
            motif=test_motifs,
            parse_method=args.parse_method,
            act_fun=args.act_fun,
            temporal_flatten=args.temporal_flatten,
            layer_by_layer=args.layer_by_layer,
            n_groups=args.n_groups,
            glob=glob,
        )

        if 'dvs' in args.dataset:
            args.channels = 2
        elif 'mnist' in args.dataset:
            args.channels = 1
        else:
            args.channels = 3
        # flops, params = profile(model, inputs=(torch.randn(1, args.channels, args.event_size, args.event_size),), verbose=False)
        # _logger.info('flops = %fM', flops / 1e6)
        # _logger.info('param size = %fM', params / 1e6)
        flops=0
        params=0
        linear_scaled_lr = args.lr * args.batch_size * args.world_size / 1024.0
        args.lr = linear_scaled_lr
        _logger.info("learning rate is %f" % linear_scaled_lr)

        if args.local_rank == 0:
            sumpram=sum([m.numel() for m in model.parameters()])
            _logger.info('Model %s created, param count: %d' %
                        (args.model, sumpram))


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
            model = nn.DataParallel(model, device_ids=devices).cuda()
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
            # checkpoint = torch.load(args.resume, map_location='cpu')
            # model.load_state_dict(checkpoint['state_dict'], False)
            resume_epoch = resume_checkpoint(
                model, args.resume,
                optimizer=None if args.no_resume_opt else optimizer,
                loss_scaler=None if args.no_resume_opt else loss_scaler,
                log_info=args.local_rank == 0)
            # print(model.get_attr('mu'))
            # print(model.get_attr('sigma'))

        if args.critical_loss or args.spike_rate:
            if args.num_gpu>1:
                model.module.set_requires_fp(True)
            else:
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
        loader_train, loader_eval, mixup_active, mixup_fn = eval('get_%s_data' % args.dataset)(
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
            num=args.cutmix_num,
            noise=args.cutmix_noise,
            num_classes=args.num_classes,
            rand_aug=args.rand_aug,
            randaug_n=args.randaug_n,
            randaug_m=args.randaug_m,
            temporal_flatten=args.temporal_flatten,
            portion=args.train_portion,
            _logger=_logger,

        )

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
            # else:
            #     load_checkpoint(model, args.eval_checkpoint, args.model_ema)
            for i in range(1):
                val_metrics,_ = validate(start_epoch, model, loader_eval, validate_loss_fn, args,arch_dir,
                                    visualize=args.visualize, spike_rate=args.spike_rate,
                                    tsne=args.tsne, conf_mat=args.conf_mat)
                print(f"Top-1 accuracy of the model is: {val_metrics['top1']:.1f}%")
            # return

        saver = None
        if args.local_rank == 0:
            decreasing = True if eval_metric == 'loss' else False

            saver = CheckpointSaver(
                model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
                checkpoint_dir=arch_dir, recovery_dir=arch_dir, decreasing=decreasing)
            with open(os.path.join(arch_dir, 'args.yaml'), 'w') as f:
                f.write(args_text)
        f=open(os.path.join(arch_dir, 'direct_genome.txt'), 'a')
        f.write(",".join(str(k) for k in genome))
        f.write('\n')
        f.close()
        try:  # train the model
            if args.reset_drop:
                model_without_ddp.reset_drop_path(0.0)
            for epoch in range(start_epoch, args.epochs):
                if epoch == 0 and args.reset_drop:
                    model_without_ddp.reset_drop_path(args.drop_path)

                if args.distributed:
                    loader_train.sampler.set_epoch(epoch)

                train_metrics = train_epoch(
                    epoch, model, loader_train, optimizer, train_loss_fn, args,_logger=_logger,
                    lr_scheduler=lr_scheduler, saver=saver, output_dir=arch_dir,
                    amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    if args.local_rank == 0:
                        _logger.info("Distributing BatchNorm running means and vars")
                    distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

                eval_metrics,_ = validate(epoch, model, loader_eval, validate_loss_fn, args, arch_dir,amp_autocast=amp_autocast,_logger=_logger,
                                        visualize=args.visualize, spike_rate=args.spike_rate,
                                        tsne=args.tsne, conf_mat=args.conf_mat)

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                    ema_eval_metrics,_ = validate(
                        epoch, model_ema.ema, loader_eval, validate_loss_fn, args, arch_dir,amp_autocast=amp_autocast, log_suffix=' (EMA)',_logger=_logger,
                        visualize=args.visualize, spike_rate=args.spike_rate,
                        tsne=args.tsne, conf_mat=args.conf_mat)
                    eval_metrics = ema_eval_metrics

                if lr_scheduler is not None:
                    # step LR for next epoch
                    lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

                update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(arch_dir, 'summary.csv'),
                    write_header=best_metric is None)

                # if saver is not None and epoch >= args.n_warm_up:
                if saver is not None:
                    # save proper checkpoint with eval metric
                    save_metric = eval_metrics[eval_metric]
                    best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
                best_metric, best_epoch = eval_metrics[eval_metric],epoch
                _logger.info('Train: {0} '.format(best_metric))

            f=open(os.path.join(arch_dir, 'direct.txt'), 'a')
            f.write(str(best_metric))
            f.write('\n')
            f.close()



        except KeyboardInterrupt:
            pass
    except MemoryError:
        return -10000, 0
    except RuntimeError:
        # return -10000, {'flops': flops / 1e6, 'param': params / 1e6}
        return -10000, 0

    # if best_metric is not None:
    #     _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


    # info=get_net_info(model)
    
    val_metrics,spikes = validate(start_epoch, model, loader_eval, validate_loss_fn, args,arch_dir,
                                    visualize=args.visualize, spike_rate=args.spike_rate,
                                    tsne=args.tsne, conf_mat=args.conf_mat,_logger=_logger,)

    return best_metric,spikes

def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,_logger,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    closses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.train()

    # t, k = adjust_surrogate_coeff(100, args.epochs)
    # model.set_attr('t', t)
    # model.set_attr('k', k)

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (inputs, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher or args.dataset != 'imnet':
            inputs, target = inputs.type(torch.FloatTensor).cuda(), target.cuda()
            if mixup_fn is not None:
                inputs, target = mixup_fn(inputs, target)
        if args.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            output = model(inputs)
            loss = loss_fn(output, target)
        if not (args.cut_mix | args.mix_up | args.event_mix) and args.dataset != 'imnet':
            # print(output.shape, target.shape)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # acc1, = accuracy(output, target)
        else:
            acc1, acc5 = torch.tensor([0.]), torch.tensor([0.])

        closs = torch.tensor([0.], device=loss.device)

        if args.critical_loss:
            closs = calc_critical_loss(model)
        loss = loss + .1 * closs

        spike_rate_avg_layer_str = ''
        threshold_str = ''



        if not args.distributed:
            losses_m.update(loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), inputs.size(0))
            top5_m.update(acc5.item(), inputs.size(0))
            closses_m.update(closs.item(), inputs.size(0))
            if args.num_gpu>1:
                spike_rate_avg_layer = model.module.get_fire_rate().tolist()
                spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in spike_rate_avg_layer]
                threshold = model.module.get_threshold()
            
            else:
                spike_rate_avg_layer = model.get_fire_rate().tolist()
                spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in spike_rate_avg_layer]
                threshold = model.get_threshold()                
            
            threshold_str = ['{:.3f}'.format(i) for i in threshold]


                  
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
                            batch_idx, len(loader),
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
                        # 'Thres: {threshold}\n'
                        # 'Mu: {mu_str}\n'
                        # 'Sigma: {sigma_str}\n'
                        .format(
                            epoch,
                            batch_idx, len(loader),
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
                            # threshold=threshold_str,
                            # mu_str=mu_str,
                            # sigma_str=sigma_str
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

    return OrderedDict([('loss', losses_m.avg)])


def validate(epoch, model, loader, loss_fn, args, arch_dir,_logger,amp_autocast=suppress,
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
            if not args.prefetcher or args.dataset != 'imnet':
                inputs = inputs.type(torch.FloatTensor).cuda()
                target = target.cuda()
            if args.channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            if not args.distributed:
                if (visualize or spike_rate or tsne or conf_mat) and not args.critical_loss:
                    if args.num_gpu>1:
                        model.module.set_requires_fp(True)
                    else:
                        model.set_requires_fp(True)

                    # if not args.critical_loss:
                    #     model.set_requires_fp(False)

            with amp_autocast():
                output = model(inputs)
            if isinstance(output, (tuple, list)):
                output = output[0]

            if not args.distributed:
                if visualize:
                    x = model.get_fp()
                    feature_path = os.path.join(arch_dir, 'feature_map')
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
                    if args.num_gpu>1:
                        avg, var, spike, avg_per_step = model.module.get_spike_info()

                    else:
                        avg, var, spike, avg_per_step = model.get_spike_info()
                    save_spike_info(
                        os.path.join(arch_dir, 'spike_info.csv'),
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
                if args.num_gpu>1:
                    spike_rate_avg_layer = model.module.get_fire_rate().tolist()
                    threshold = model.module.get_threshold()
                    threshold_str = ['{:.3f}'.format(i) for i in threshold]
                    spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in spike_rate_avg_layer]
                    tot_spike = model.module.get_tot_spike()
                else:
                    spike_rate_avg_layer = model.get_fire_rate().tolist()
                    threshold = model.get_threshold()
                    threshold_str = ['{:.3f}'.format(i) for i in threshold]
                    spike_rate_avg_layer_str = ['{:.3f}'.format(i) for i in spike_rate_avg_layer]
                    tot_spike = model.get_tot_spike()                    

            if args.critical_loss:
                closs = calc_critical_loss(model)
            loss = loss + .1 * closs

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
            plot_tsne(feature_vec, feature_cls, os.path.join(arch_dir, 't-sne-2d.eps'))
            plot_tsne_3d(feature_vec, feature_cls, os.path.join(arch_dir, 't-sne-3d.eps'))
        if conf_mat:
            logits_vec = torch.cat(logits_vec)
            labels_vec = torch.cat(labels_vec)
            plot_confusion_matrix(logits_vec, labels_vec, os.path.join(arch_dir, 'confusion_matrix.eps'))

    return metrics,tot_spike



def get_net_info(args, gen,genome,ms):
    """
    Modified from https://github.com/mit-han-lab/once-for-all/blob/
    35ddcb9ca30905829480770a6a282d49685aa282/ofa/imagenet_codebase/utils/pytorch_utils.py#L139
    """
    from ofa.imagenet_codebase.utils.pytorch_utils import count_parameters, measure_net_latency

    # artificial input data

    if args.bns:
        from cellmodel import NetworkCIFAR
    else:
        from cell123model import NetworkCIFAR
    

    test_motifs,ids = micro_encoding.decode_motif(args.layers*ms,args.bits,genome.astype(int))
    net = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            dataset=args.dataset,
            step=args.step,
            encode_type=args.encode,
            node_type=eval(args.node_type),
            threshold=args.threshold,
            tau=args.tau,
            sigmoid_thres=args.sigmoid_thres,
            requires_thres_grad=args.requires_thres_grad,
            spike_output=not args.no_spike_output,
            C=args.init_channels,
            layers=args.layers*ms,
            auxiliary=args.auxiliary,
            motif=test_motifs,
            parse_method=args.parse_method,
            act_fun=args.act_fun,
            temporal_flatten=args.temporal_flatten,
            layer_by_layer=args.layer_by_layer,
            n_groups=args.n_groups,
            cell_type=genome[-1],
        )

    if 'dvs' in args.dataset:
        args.channels = 2
    elif 'mnist' in args.dataset:
        args.channels = 1
    else:
        args.channels = 3
    inputs = torch.randn(1, args.channels, 224, 224)


    # move network to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        net = net.to(device)
        cudnn.benchmark = True
        inputs = inputs.to(device)

    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # parameters
    net_info['params'] = count_parameters(net)

    # flops
    net_info['flops'] = int(profile_macs(copy.deepcopy(net), inputs))


    return net_info
