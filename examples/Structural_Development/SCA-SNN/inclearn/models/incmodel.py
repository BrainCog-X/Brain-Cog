import numpy as np
import random
import time
import math
import os
from copy import deepcopy
from scipy.spatial.distance import cdist
from torchvision.utils import save_image

import torch
# import pdb
from torch.nn import DataParallel
from torch.nn import functional as F
from torch import nn



from inclearn.convnet import network
from inclearn.models.base import IncrementalLearner
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter
from inclearn.tools.memory import MemorySize
from inclearn.tools.scheduler import GradualWarmupScheduler
from inclearn.convnet.utils import extract_features, update_classes_mean, finetune_last_layer

# Constants
EPSILON = 1e-8


class IncModel(IncrementalLearner):
    def __init__(self, cfg, trial_i, _run, ex, tensorboard, inc_dataset):
        super().__init__()
        self._cfg = cfg
        self._device = cfg['device']
        self._ex = ex
        self._run = _run  # the sacred _run object.

        # Data
        self._inc_dataset = inc_dataset
        self._n_classes = 0
        self.classnum_list = []
        self.sample_list = []
        self._trial_i = trial_i  # which class order is used

        # Optimizer paras
        self._opt_name = cfg["optimizer"]
        self._warmup = cfg['warmup']
        self._lr = cfg["lr"]
        self._weight_decay = cfg["weight_decay"]
        self._n_epochs = cfg["epochs"]
        self._scheduling = cfg["scheduling"]
        self._lr_decay = cfg["lr_decay"]
        self.torc=cfg['distillation']
        self.prune = cfg.get('prune', False)


        # Logging
        self._tensorboard = tensorboard
        if f"trial{self._trial_i}" not in self._run.info:
            self._run.info[f"trial{self._trial_i}"] = {}
        self._val_per_n_epoch = cfg["val_per_n_epoch"]

        # Model
        self._dea = cfg['dea']  # Whether to expand the representation
        self._network = network.BasicNet(
            cfg["convnet"],
            cfg=cfg,
            nf=cfg["channel"],
            device=self._device,
            use_bias=cfg["use_bias"],
            dataset=cfg["dataset"],
        )


        if self._cfg.get("caculate_params", False):
            self._parallel_network = self._network
        else:
            # 并行计算
            # gpus = [0, 1, 2, 3]
            # self._parallel_network = DataParallel(self._network, device_ids=gpus, output_device=gpus[0])
            self._parallel_network = DataParallel(self._network)

        self._train_head = cfg["train_head"]
        self._infer_head = cfg["infer_head"]
        self._old_model = None

        # Learning
        self._temperature = cfg["temperature"]
        self._distillation = cfg["distillation"]
        self.lamb = cfg["distlamb"]

        # Memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], inc_dataset, cfg["memory_size"],
                                       cfg["fixed_memory_per_cls"])
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]

        if self._cfg["save_ckpt"]:
            save_path = os.path.join(os.getcwd(), f"{self._cfg.exp.saveckpt}")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), f"{self._cfg.exp.saveckpt}/mem")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

    def eval(self):
        self._parallel_network.eval()

    def train(self):
        if self._dea:
            self._parallel_network.train()
            self._parallel_network.module.convnets[-1].train()
            if self._task >= 1:
                for i in range(self._task):
                    self._parallel_network.module.convnets[i].eval()
        else:
            self._parallel_network.train()

    def _before_task(self, taski, inc_dataset,mask,min_dist,all_dist):
        self._ex.logger.info(f"Begin step {taski}")

        # Update Task info
        self._task = taski
        self._n_classes += self._task_size
        self.classnum_list.append(self._task_size)
        self.sample_list = [ int(2000/(self._n_classes-10)) for i in range(self._n_classes-10)] + [ 500 for i in range(10)]

        # Memory
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        self._ex.logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._network.add_classes(self._task_size,min_dist)
        self._network.task_size = self._task_size
        mask.model=self._network.convnets[-1]
        mask.init_length(taski,task_nn=self._network.task_nn)
        self.set_optimizer()

    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        self._ex.logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

        # if self._dea and self._task > 0 and not self._cfg.get("caculate_params", False):
        #     for i in range(self._task):
        #         for p in self._parallel_network.module.convnets[i].parameters():
        #             p.requires_grad = False

        self._optimizer = factory.get_optimizer(self._network.convnets[-1].parameters(),
                                                self._opt_name, lr, weight_decay)

        if "cos" in self._cfg["scheduler"]:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._n_epochs)
        else:
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   self._scheduling,
                                                                   gamma=self._lr_decay)

        if self._warmup:
            print("warmup")
            self._warmup_scheduler = GradualWarmupScheduler(self._optimizer,
                                                            multiplier=1,
                                                            total_epoch=self._cfg['warmup_epochs'],
                                                            after_scheduler=self._scheduler)

    def _train_task(self, task_i,train_loader, val_loader,mask,min_dist,all_dist):

        self._ex.logger.info(f"nb {len(train_loader.dataset)}")

        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])
        train_new_accu = ClassErrorMeter(accuracy=True)
        train_old_accu = ClassErrorMeter(accuracy=True)

        self._optimizer.zero_grad()
        self._optimizer.step()

        for epoch in range(self._n_epochs):
            # torch.cuda.empty_cache()
            _loss, _loss_div, _loss_trip, _loss_dist, _loss_atmap = 0.0, 0.0, 0.0, 0.0, 0.0
            accu.reset()
            train_new_accu.reset()
            train_old_accu.reset()
            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    if self.torc:
                        self._network.convnets[-1].classifer.reset_parameters()
                        if self._cfg['use_div_cls']:
                            self._network.convnets[-1].aux_classifier.reset_parameters()
                    else:
                        self._network.convnets[task_i].classifer.reset_parameters()
                        if self._cfg['use_div_cls']:
                            self._network.aux_classifier[task_i].reset_parameters()
            for i, (inputs, targets) in enumerate(train_loader, start=1):
                self.train()
                self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                loss_ce, loss_div, loss_trip, loss_dist, loss_atmap = self._forward_loss(
                    task_i,
                    inputs,
                    targets,
                    old_classes,
                    new_classes,
                    epoch,
                    accu=accu,
                    new_accu=train_new_accu,
                    old_accu=train_old_accu,
                    mask=mask
                )

                loss = loss_ce

                if self._cfg["distillation"] and self._task > 0:
                    # trade-off - the lambda from the paper if lamb=-1
                    if self.lamb == -1:
                        lamb = (self._n_classes - self._task_size) / self._n_classes
                        loss = (1-lamb) * loss + lamb * loss_dist
                    else:
                        loss =  loss + self.lamb * loss_dist

                if self._cfg["use_div_cls"] and self._task > 0:
                    loss += loss_div           


                loss.backward()
                self._optimizer.step()

                if self.torc:
                    if self._cfg["postprocessor"]["enable"]:
                        if self._cfg["postprocessor"]["type"].lower() == "cr" or self._cfg["postprocessor"]["type"].lower() == "aver":
                            for p in self._network.convnets[-1].classifer.parameters():
                                p.data.clamp_(0.0)

                _loss += loss_ce
                _loss_trip += loss_trip
                _loss_div += loss_div
                _loss_dist += loss_dist
                _loss_atmap += loss_atmap 
            
            if task_i>0:
                mask.init_mask(self._task,epoch,dim_cur=self._network.channel_dim, task_nn=self._network.task_nn,all_dist=all_dist,all_model=self._network.convnets)
                mat=mask.do_mask(self._task)
                

            _loss = _loss.item()
            _loss_div = _loss_div.item()
            _loss_trip = _loss_trip.item()
            _loss_dist = _loss_dist.item()
            _loss_atmap = _loss_atmap.item()
            if not self._warmup:
                self._scheduler.step()
            self._ex.logger.info(
                "Task {}/{}, Epoch {}/{} => Clf loss: {} Div loss: {}, Knowledge Distllation loss:{}, Train Accu: {}, Train@5 Acc: {}".
                format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_loss / i, 3),
                    round(_loss_div / i, 3),
                    round(_loss_dist / i, 3),
                    round(accu.value()[0], 3),
                    round(accu.value()[1], 3),
                ))

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

        if self.torc:
            # For the large-scale dataset, we manage the data in the shared memory.
            self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory

            utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")
            utils.display_feature_norm(task_i,self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                    self._increments, "Trainset",mask=mask)
            self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)

    def _forward_loss(self, task_i,inputs, targets, old_classes, new_classes, epoch, accu=None, new_accu=None, old_accu=None,mask=None):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

        outputs = self._parallel_network(task_i,inputs,mask)
        if accu is not None:
            accu.add(outputs['logit'], targets)
        return self._compute_loss(task_i, inputs, targets, outputs, old_classes, new_classes, epoch,mask=mask)

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def hcl(self, fstudent, fteacher, targets):
        loss_all = 0.0
        fs = fstudent
        select_teacher =  self._cfg.get("select_teacher",False)

        if select_teacher:
            for i in range(len(fteacher)):
                ft = fteacher[i]
                if i > 0:
                    old_classes = np.logical_and((targets < (self._n_classes - self._task_size * (len(fteacher)-i))).cpu(), (targets >= (self._n_classes - self._task_size * (len(fteacher)-i+1))).cpu())
                else:
                    old_classes = (targets < (self._n_classes - self._task_size * len(fteacher))).cpu()
                classes_indice = torch.from_numpy(np.where(old_classes==True)[0]).to(self._device)
                # targets_old = torch.index_select(targets_old, 0, old_classes_indice)
                # log_probs_new = torch.index_select(log_probs_new, 0, old_classes_indice)
                fs = torch.index_select(fstudent, 0, classes_indice)
                ft = torch.index_select(ft, 0, classes_indice) 
                n,c,h,w = fs.shape
                if n == 0:
                    break
                loss = F.mse_loss(fs, ft, reduction='mean')
                cnt = 1.0
                tot = 1.0
                for l in [4,2,1]:
                    if l >=h:
                        continue
                    tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                    tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                    cnt /= 2.0
                    loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                    tot += cnt
                loss = loss / tot
                loss_all = loss_all + loss               
        else:
            for i in range(len(fteacher)):
                ft = fteacher[i]
                n,c,h,w = fs.shape
                if n == 0:
                    break
                loss = F.mse_loss(fs, ft, reduction='mean')
                cnt = 1.0
                tot = 1.0
                for l in [4,2,1]:
                    if l >=h:
                        continue
                    tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
                    tmpft = F.adaptive_avg_pool2d(ft, (l,l))
                    cnt /= 2.0
                    loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                    tot += cnt
                loss = loss / tot
                loss_all = loss_all + loss
        return loss_all

    def _compute_loss(self, task_i, inputs, targets, outputs, old_classes, new_classes, epoch,mask=None):

        loss = F.cross_entropy(outputs['logit'], targets)

        trip_loss = torch.zeros([1]).cuda()

        atmap_loss = torch.zeros([1]).cuda()

        if outputs['div_logit'] is not None:
            div_targets = targets.clone()
            if self._cfg["div_type"] == "n+1":
                div_targets[old_classes] = 0
                div_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - 1
            elif self._cfg["div_type"] == "1+1":
                div_targets[old_classes] = 0
                div_targets[new_classes] = 1
            elif self._cfg["div_type"] == "n+t":
                div_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - self._task
                for i in range(self._task):
                    if i > 0:
                        old_class = np.logical_and((targets < (self._n_classes - self._task_size * (self._task-i))).cpu(), (targets >= (self._n_classes - self._task_size * (self._task-i+1))).cpu())
                    else:
                        old_class = (targets < (self._n_classes - self._task_size * self._task)).cpu()

                    div_targets[old_class] = i
            # import pdb
            # pdb.set_trace()
            div_loss = F.cross_entropy(outputs['div_logit'], div_targets)

        else:
            div_loss = torch.zeros([1]).cuda()   

        if self._cfg["distillation"] and self._old_model is not None:
            outputs_old = self._old_model(task_i-1,inputs,mask=None)
            targets_old = outputs_old['logit'].detach()

            if self._cfg["disttype"] == "KL":
                log_probs_new = (outputs['logit'][:, :-self._task_size] / self._temperature).log_softmax(dim=1)
                if self._task > 1 and self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "aver":
                        targets_old = self._old_model.module.postprocessor.post_process(targets_old, self._task_size, self.classnum_list[:-1], self._task-1)
                    else:
                        targets_old = self._old_model.module.postprocessor.post_process(targets_old, self._task_size)
                modify =  self._cfg.get("modify_new",False)
                if modify:
                    old_weight_norm = torch.norm(self._network.convnets[-1].classifer.weight[:-self._task_size], p=2, dim=1)
                    new_weight_norm = torch.norm(self._network.convnets[-1].classifer.weight[-self._task_size:], p=2, dim=1)
                    gamma = old_weight_norm.mean() / new_weight_norm.mean()
         
                    targets_old[new_classes,:] = targets_old[new_classes,:] * gamma
                probs_old = (targets_old / self._temperature).softmax(dim=1)
                
                dist_loss = F.kl_div(log_probs_new, probs_old, reduction="batchmean")
            
                
            else:
                dist_loss = self.cross_entropy(outputs['logit'][:, :-self._task_size], targets_old, exp=1.0 / self._temperature)
        else:
            dist_loss = torch.zeros([1]).cuda()         

        return loss, div_loss, trip_loss, dist_loss, atmap_loss

    def _after_task(self, taski, inc_dataset,mask=None):
        network = deepcopy(self._parallel_network)
        network.eval()
        
        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"] and not self.prune:
            self._ex.logger.info("save model")
            save_path = os.path.join(os.getcwd(), f"{self._cfg.exp.saveckpt}")
            torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(save_path, self._task))

        if self.torc:
            if self._cfg["postprocessor"]["enable"]:
                self._update_postprocessor(taski,inc_dataset,mask=mask)

        if self._cfg["infer_head"] == 'NCM':
            self._ex.logger.info("compute prototype")
            self.update_prototype()

        if self._memory_size.memsize != 0:
            self._ex.logger.info("build memory")
            self.build_exemplars(taski,inc_dataset, self._coreset_strategy,mask=mask)

            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), f"{self._cfg.exp.saveckpt}/mem")
                memory = {
                    'x': inc_dataset.data_memory,
                    'y': inc_dataset.targets_memory,
                    'herding': self._herding_matrix
                }
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not (os.path.exists(f"{save_path}/mem_step{self._task}.ckpt") and self._cfg['load_mem']):
                    torch.save(memory, "{}/mem_step{}.ckpt".format(save_path, self._task))
                    self._ex.logger.info(f"Save step{self._task} memory!")

        # utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")

        
        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        if not self._cfg.get("caculate_params", False):
            self._old_model.module.freeze()
        del self._inc_dataset.shared_data_inc
        self._inc_dataset.shared_data_inc = None

    def _eval_task(self,task_i, data_loader,mask):
        # if self._cfg.get("caculate_params", False):
            # from thop import profile
            # self._parallel_network.eval()
            # with torch.no_grad():
            #     input = torch.randn(1, 3, 256, 256).to(self._device, non_blocking=True)
            #     flops, params = profile(self._parallel_network, inputs=(input,))
            # ypred = flops/1000**3
            # ytrue = params/1000**2
            # from torchstat import stat
            # stat(self._parallel_network, (3, 256, 256))
            # ypred,ytrue = 0,0
        # else:
        if self._infer_head == "softmax":
            ypred, ytrue = self._compute_accuracy_by_netout(task_i,data_loader,mask)
        elif self._infer_head == "NCM":
            ypred, ytrue = self._compute_accuracy_by_ncm(data_loader)
        else:
            raise ValueError()

        return ypred, ytrue


    def _compute_accuracy_by_netout(self, task_i,data_loader,mask):
        preds, targets = [], []
        self._parallel_network.eval()
        if self._cfg.get("caculate_params", False):
            with torch.no_grad():
                from thop import profile
                inputs = torch.randn(1, 3, 112, 112)
                flops, params = profile(self._parallel_network, (inputs,))
                preds = flops/1000**3
                targets = params/1000**2
                # print('flops: ', flops, 'params: ', params)
                # for i, (inputs, lbls) in enumerate(data_loader):
                #     from thop import profile
                #     # inputs = inputs.to(self._device, non_blocking=True)
                    
                #     flops, params = profile(self._parallel_network, inputs[0])
                #     preds = flops/1000**3
                #     targets = params/1000**2
                #     break                             
        else:
            with torch.no_grad():            
                for i, (inputs, lbls) in enumerate(data_loader):
                    inputs = inputs.to(self._device, non_blocking=True)
                    _preds = self._parallel_network(task_i,inputs,mask)['logit']
                    if self.torc:
                        if self._cfg["postprocessor"]["enable"] and self._task > 0:
                            if self._cfg["postprocessor"]["type"].lower() == "aver":
                                _preds = self._network.postprocessor.post_process(_preds, self._task_size, self.classnum_list, self._task)
                            else:
                                _preds = self._network.postprocessor.post_process(_preds, self._task_size)
                    preds.append(_preds.detach().cpu().numpy())
                    targets.append(lbls.long().cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
        return preds, targets

    def _compute_accuracy_by_ncm(self, loader):
        features, targets_ = extract_features(self._parallel_network, loader)
        targets = np.zeros((targets_.shape[0], self._n_classes), np.float32)
        targets[range(len(targets_)), targets_.astype("int32")] = 1.0

        class_means = (self._class_means.T / (np.linalg.norm(self._class_means.T, axis=0) + EPSILON)).T

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
        # Compute score for iCaRL
        sqd = cdist(class_means, features, "sqeuclidean")
        score_icarl = (-sqd).T
        return score_icarl[:, :self._n_classes], targets_

    def _update_postprocessor(self, taski,inc_dataset,mask=None):
        if self._cfg["postprocessor"]["type"].lower() == "bic":
            if False:#self._cfg["postprocessor"]["disalign_resample"] is True:
                bic_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                     inc_dataset.targets_inc,
                                                     mode="train",
                                                     resample='disalign_resample')
            else:
                xdata, ydata = inc_dataset._select(inc_dataset.data_train,
                                                   inc_dataset.targets_train,
                                                   low_range=0,
                                                   high_range=self._n_classes)
                bic_loader = inc_dataset._get_loader(xdata, ydata, shuffle=True, mode='train')
            bic_loss = None
            self._network.postprocessor.reset(n_classes=self._n_classes)
            self._network.postprocessor.update(self._ex.logger,
                                               self._task_size,
                                               self._parallel_network,
                                               bic_loader,
                                               loss_criterion=bic_loss,
                                               taski=taski,
                                               mask=mask)
        elif self._cfg["postprocessor"]["type"].lower() == "cr":
            self._ex.logger.info("Post processor cr update !")
            self._network.postprocessor.update(self._network.convnets[-1].classifer, self._task_size)
        elif self._cfg["postprocessor"]["type"].lower() == "aver":
            self._ex.logger.info("Post processor aver update !")
            self._network.postprocessor.update(self._network.convnets[-1].classifer, self._task_size, self.classnum_list, self._task)

    def update_prototype(self):
        if hasattr(self._inc_dataset, 'shared_data_inc'):
            shared_data_inc = self._inc_dataset.shared_data_inc
        else:
            shared_data_inc = None
        self._class_means = update_classes_mean(self._parallel_network,
                                                self._inc_dataset,
                                                self._n_classes,
                                                self._task_size,
                                                share_memory=self._inc_dataset.shared_data_inc,
                                                metric='None')

    def build_exemplars(self, task_i,inc_dataset, coreset_strategy,mask=None):
        save_path = os.path.join(os.getcwd(), f"{self._cfg.exp.saveckpt}/mem/mem_step{self._task}.ckpt")
        if self._cfg["load_mem"] and os.path.exists(save_path):
            memory_states = torch.load(save_path)
            self._inc_dataset.data_memory = memory_states['x']
            self._inc_dataset.targets_memory = memory_states['y']
            self._herding_matrix = memory_states['herding']
            self._ex.logger.info(f"Load saved step{self._task} memory!")
            return

        if coreset_strategy == "random":
            from inclearn.tools.memory import random_selection

            self._inc_dataset.data_memory, self._inc_dataset.targets_memory = random_selection(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._ex.logger,
                inc_dataset,
                self._memory_per_class,
            )
        elif coreset_strategy == "iCaRL":
            from inclearn.tools.memory import herding
            data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None else self._inc_dataset.data_inc
            self._inc_dataset.data_memory, self._inc_dataset.targets_memory, self._herding_matrix = herding(
                task_i,
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._herding_matrix,
                inc_dataset,
                data_inc,
                self._memory_per_class,
                self._ex.logger,
                mask=mask
            )
        else:
            raise ValueError()

    def validate(self, data_loader):
        if self._infer_head == 'NCM':
            self.update_prototype()
        ypred, ytrue = self._eval_task(data_loader)
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        self._ex.logger.info(f"test top1acc:{test_acc_stats['top1']}")
        return test_acc_stats['top1']['total']

    def after_prune(self, taski, inc_dataset):
        x = torch.randn(1, 3, 32, 32)
        self._network = self._network.cpu()
        dim1, dim2 = self._network.caculate_dim(x)
        del self._network.classifier
        self._network.classifier = self._network._gen_classifier(dim1, self._n_classes)
        
        if self._network.se is not None:
            del self._network.se
            ft_type = self._cfg.get('feature_type', 'ce')
            at_res = self._cfg.get('attention_use_residual', False)
            self._network.se = factory.get_attention(dim1, ft_type, at_res)
        if taski > 0:
            del self._network.aux_classifier
            self._network.aux_classifier = self._network._gen_classifier(dim2, self._task_size+1)

        del self._parallel_network
        self._parallel_network = DataParallel(self._network)


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="sum") * (self.T**2) / y_s.shape[0]
        # loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2) / y_s.shape[0]
        
        return loss
