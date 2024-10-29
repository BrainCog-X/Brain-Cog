import sys
import os
import os.path as osp
import copy
import time
import shutil
import cProfile
import logging
from pathlib import Path
import numpy as np
import random
from easydict import EasyDict as edict
from tensorboardX import SummaryWriter
import os
import inclearn.convnet.maskcl2 as Mask

os.environ['CUDA_VISIBLE_DEVICES']='0'

repo_name = 'TCIL'
base_dir = '/data1/hanbing/TCIL10/'
sys.path.insert(0, base_dir)

from sacred import Experiment
ex = Experiment(base_dir=base_dir, save_git_info=False)


import torch

from inclearn.tools import factory, results_utils, utils

from inclearn.tools.metrics import IncConfusionMeter
from inclearn.tools.similar import Appr



def initialization(config, seed, mode, exp_id):

    torch.backends.cudnn.benchmark = True  # This will result in non-deterministic results.
    # ex.captured_out_filter = lambda text: 'Output capturing turned off.'
    cfg = edict(config)
    utils.set_seed(cfg['seed'])
    if exp_id is None:
        exp_id = -1
        cfg.exp.savedir = "./logs_aphal"
    logger = utils.make_logger(str(exp_id)+str(cfg.exp.name)+str(mode), savedir=cfg.exp.savedir)

    # Tensorboard
    exp_name = '{exp_id}_{cfg["exp"]["name"]}' if exp_id is not None else '../inbox/{cfg["exp"]["name"]}'
    tensorboard_dir = cfg["exp"]["tensorboard_dir"] + "/{exp_name}"

    # If not only save latest tensorboard log.
    # if Path(tensorboard_dir).exists():
    #     shutil.move(tensorboard_dir, cfg["exp"]["tensorboard_dir"] + f"/../inbox/{time.time()}_{exp_name}")

    tensorboard = SummaryWriter(tensorboard_dir)

    return cfg, logger, tensorboard


@ex.command
def train(_run, _rnd, _seed):
    cfg, ex.logger, tensorboard = initialization(_run.config, _seed, "train", _run._id)
    ex.logger.info(cfg)
    cfg.data_folder = osp.join(base_dir, "data")

    start_time = time.time()
    _train(cfg, _run, ex, tensorboard)
    ex.logger.info("Training finished in {}s.".format(int(time.time() - start_time)))


def _train(cfg, _run, ex, tensorboard):
    device = factory.set_device(cfg)
    trial_i = cfg['trial']
    torc=cfg['distillation']

    inc_dataset = factory.get_data(cfg, trial_i)
    ex.logger.info("classes_order")
    ex.logger.info(inc_dataset.class_order)

    model = factory.get_model(cfg, trial_i, _run, ex, tensorboard, inc_dataset)
    mask=Mask.Mask(model._network.convnets[-1])
    if _run.meta_info["options"]["--file_storage"] is not None:
        _save_dir = osp.join(_run.meta_info["options"]["--file_storage"], str(_run._id))
    else:
        _save_dir = cfg["exp"]["ckptdir"]

    results = results_utils.get_template_results(cfg)
    appr=Appr(model._network,10,torc)
    for task_i in range(inc_dataset.n_tasks):
        task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(task_i)

        model.set_task_info(
            task=task_info["task"],
            total_n_classes=task_info["max_class"],
            increment=task_info["increment"],
            n_train_data=task_info["n_train_data"],
            n_test_data=task_info["n_test_data"],
            n_tasks=inc_dataset.n_tasks,
        )
        if torc:
            strategy, expert_id ,min_dist,all_dist,all_dist2= appr.learn(task_i, val_loader,  cfg['batch_size'],device)
        else:
            strategy, expert_id ,min_dist,all_dist,all_dist2= appr.learn(task_i, test_loader[task_i],  cfg['batch_size'],device)
        print("Task:",task_i,strategy, expert_id,min_dist,all_dist,all_dist2)


        model.before_task(task_i, inc_dataset,mask,min_dist,all_dist)
        
        # TODO: Move to incmodel.py
        if 'min_class' in task_info:
            ex.logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))

        if torc:
            model.train_task(task_i,train_loader, test_loader,mask,min_dist,all_dist)
            model.after_task(task_i, inc_dataset,mask)
            appr.after_learn(task_i, val_loader, cfg['batch_size'],device)

        else:
            model.train_task(task_i,train_loader, val_loader[task_i],mask,min_dist,all_dist)
            appr.after_learn(task_i, test_loader[task_i], cfg['batch_size'],device)


        if torc:
            ex.logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))
            ypred, ytrue = model.eval_task(task_i,test_loader,mask)
            acc_stats = utils.compute_accuracy(ypred, ytrue, increments=model._increments, n_classes=model._n_classes)
            #Logging
            model._tensorboard.add_scalar("taskaccu/trial{trial_i}", acc_stats["top1"]["total"], task_i)
            _run.log_scalar("trial{trial_i}_taskaccu", acc_stats["top1"]["total"], task_i)
            _run.log_scalar("trial{trial_i}_task_top5_accu", acc_stats["top5"]["total"], task_i)
            ex.logger.info("top1:"+str(acc_stats['top1']))
            ex.logger.info("top5:"+str(acc_stats['top5']))
            results["results"].append(acc_stats)
        else:
            for taski in range(task_i+1):
                ypred, ytrue = model.eval_task(taski,test_loader[taski],mask)
    
                acc_stats = utils.compute_accuracy(ypred, ytrue, increments=[1], n_classes=model._n_classes)

                model._tensorboard.add_scalar(f"taskaccu/trial{trial_i}", acc_stats["top1"]["total"], taski)

                _run.log_scalar(f"trial{trial_i}_taskaccu", acc_stats["top1"]["total"], taski)
                _run.log_scalar(f"trial{trial_i}_task_top5_accu", acc_stats["top5"]["total"], taski)

                ex.logger.info(f"top1:{acc_stats['top1']}")
                ex.logger.info(f"top5:{acc_stats['top5']}")

                results["results"].append(acc_stats)

    top1_avg_acc, top5_avg_acc = results_utils.compute_avg_inc_acc(results["results"])

    _run.info["trial{trial_i}"]["avg_incremental_accu_top1"] = top1_avg_acc
    _run.info["trial{trial_i}"]["avg_incremental_accu_top5"] = top5_avg_acc
    ex.logger.info("Average Incremental Accuracy Top 1: {} Top 5: {}.".format(
        _run.info["trial{trial_i}"]["avg_incremental_accu_top1"],
        _run.info["trial{trial_i}"]["avg_incremental_accu_top5"],
    ))
    if cfg["exp"]["name"]:
        results_utils.save_results(results, cfg["exp"]["name"])


if __name__ == "__main__":
    ex.add_config("/data1/hanbing/SCA-SNN/configs/train.yaml")
    ex.run_commandline()
