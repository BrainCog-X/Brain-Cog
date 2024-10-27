import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F

class Appr(object):
    """ Class implementing the TALL """
    def __init__(self, pretrained_feat_extractor, num_task, device='cuda', args=None):

        self.task2expert = []
        self.expert2task = []
        
        self.task2mean = {}
        self.task2cov = {}
        for i in range(num_task):
            self.task2mean[i]=[]
            self.task2cov[i]=[]
        self.task_dist = torch.zeros(num_task, num_task).to(device='cuda')
        self.task_dist2 = torch.zeros(num_task, num_task).to(device='cuda')
        self.feat_extractor = pretrained_feat_extractor
        self.task_relatedness_method = "mean"
        self.reuse_threshold=0.3
        self.reuse_cell_threshold=0.75

    def get_mean_cov_feats(self,  taski, data, device):
        """compute mean and cov for features of data extracted by the expert of task t

        """
        # data = deepcopy(data)  # copy for using different preprocess
        
        # self.model.requires_grad_(False)
        # self.model.eval()
        # self.model.set_current_task(t)
        steps=10
        class_num = steps*taski
        labels = torch.arange(class_num,class_num+steps).view(-1, 1).to(device)
        all_task_feats={}
        for t in range(taski):
            all_task_feats[t]=[[] for _ in range(steps)]

        self.feat_extractor.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data):
                x, y = x.to(device), y.to(device)
                index = labels == y.view(1, -1) # CxC
                for t_p in range(taski):
                    feat = self.feat_extractor(t_p,x,mask=None,classify=False)['features']
                    feat = feat.view(feat.size(0), -1)

                    for i in range(steps):
                         all_task_feats[t_p][i].append(feat[index[i]])
            
            
            feat_means = {}
            feat_covs = {}
            all_feats_cat={}
            for t_p in range(taski):
                feat_means[t_p] = []
                feat_covs [t_p]= []
                all_feats_cat[t_p] = [torch.cat(feats, axis=0) for feats in all_task_feats[t_p]]

                for feat in all_feats_cat[t_p]:
                    feat_mean, feat_cov = gaussian_mean_cov(feat)
                    feat_means[t_p].append(feat_mean)
                    # feat_covs[t_p].append(feat_cov)
            # feat_means = [torch.mean(feat, dim=0) for feat in all_feats_cat]

        return feat_means, feat_covs, all_feats_cat
    
        
    def after_get_mean_cov_feats(self, taski, data, device):
        """compute mean and cov for features of data extracted by the expert of task t

        """
        # data = deepcopy(data)  # copy for using different preprocess
        
        # self.model.requires_grad_(False)
        # self.model.eval()
        # self.model.set_current_task(t)
        steps=10
        class_num = steps*taski
        labels = torch.arange(class_num,class_num+steps).view(-1, 1).to(device)
        all_task_feats=[[] for _ in range(steps)]
        self.feat_extractor.eval()
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data):
                x, y = x.to(device), y.to(device)
                # forward
                feat = self.feat_extractor(taski,x,mask=None,classify=False)['features']
                feat = feat.view(feat.size(0), -1)

                index = labels == y.view(1, -1) # CxC
                for i in range(steps):
                    all_task_feats[i].append(feat[index[i]])
            
            feat_means= []
            feat_covs = []
            all_feats_cat= [torch.cat(feats, axis=0) for feats in all_task_feats]

            for feat in all_feats_cat:
                feat_mean, feat_cov = gaussian_mean_cov(feat)
                feat_means.append(feat_mean)
                # feat_covs.append(feat_cov)
            # feat_means = [torch.mean(feat, dim=0) for feat in all_feats_cat]

        return feat_means, feat_covs, all_feats_cat

    def add_mean_cov(self, taski,mean, cov=None):
        self.task2mean[taski].append(mean)
        # self.task2cov[taski].append(cov)

    def task_relatedness_knnkl(self, task_id, p_task_id, all_feats):
        """
        Params:
            all_feat: shape C x N_c x D
        """
        # means and features of current data from expert of p_task_id
        # feat_means, all_feats = means_and_feats[p_task_id]
        # means of data of p_task_id
        p_feat_means = self.task2mean[p_task_id][p_task_id] # C x D
        feat_means = self.task2mean[task_id][p_task_id] # C x D
        # p_feat_cov = self.task2cov[p_task_id][p_task_id] # C x D
        # feat_cov = self.task2cov[task_id][p_task_id] # C x D
        p_feat_means = torch.stack(p_feat_means, dim=0)

        task_dist = 0
        task_dist2=0
        d = p_feat_means.shape[-1]
        n = 0
        flag = False
        for i in range(len(feat_means)):
            # for each current class
            n += all_feats[i].shape[0]

            dist_in = all_feats[i] - feat_means[i] # N_c x D
            dist_in = torch.sqrt(torch.sum(dist_in ** 2, dim=-1)) # N_c
            
            # N_c x C x D
            dist_out = torch.unsqueeze(all_feats[i], dim=1) - torch.unsqueeze(p_feat_means, dim=0)
            dist_out, _ = torch.min(torch.sqrt(torch.sum(dist_out ** 2, dim=-1)), dim=-1)  # N_c

            dist = torch.mean(torch.log(dist_out / (0.9*dist_in)))
            #dist = torch.mean(torch.log(dist_out))

            # if dist <= 0:
            #     flag = True

            task_dist += torch.maximum(dist, torch.zeros_like(dist))
            task_dist2 += torch.maximum(dist, torch.zeros_like(dist))
        
        # task_dist = task_dist / len(feat_means)
        # task_dist = 1 - torch.exp(-2*task_dist)
        task_dist = torch.minimum(1 - torch.exp(-2*task_dist), task_dist)

        if task_dist == 0:
            task_dist = torch.ones_like(task_dist)

        return task_dist,task_dist2

    def get_relatedness(self, task_id, feats):
        """Compute relatedness
        
        """
        # the distance between task_id and task_id 
        self.task_dist[task_id][task_id] = 0
        self.task_dist2[task_id][task_id] = 0

        for p_task_id in range(task_id):
        # for p_task_id in range(task_id + 1):
            # task_dist = self.task_relatedness_cos(task_id, p_task_id)
            task_dist,task_dist2 = self.task_relatedness_knnkl(task_id, p_task_id, feats[p_task_id])
            # task_dist = self.task_relatedness_CKA(task_id, p_task_id, feats)
            # task_dist = self.task_relatedness_gaussian_kl(task_id, p_task_id)
            self.task_dist[task_id][p_task_id] = task_dist
            self.task_dist[p_task_id][task_id] = task_dist
            self.task_dist2[task_id][p_task_id] = task_dist2
            self.task_dist2[p_task_id][task_id] = task_dist2
    
    def strategy(self, task_id, num_train_samples):
        """ Find the expert to be reused. If not found, return -1.
        
        """
        expert = -1
        min_dist = None

        
        all_dist = []
        all_dist2 = []
        for expert_id, p_tasks in enumerate(self.expert2task):
            d = self.task_dist[task_id, p_tasks] 
            dd=  self.task_dist2[task_id, p_tasks] 
            if self.task_relatedness_method == "mean":
                s = torch.mean(d).item()
                ss = torch.mean(dd).item()
            elif self.task_relatedness_method == "max":
                s = torch.max(d).item()
            elif self.task_relatedness_method == "min":
                s = torch.min(d).item()
            else:
                raise Exception("Unknown reuse strategy !!!")
            all_dist.append(s)
            all_dist2.append(ss)
            if min_dist is None:
                min_dist = s
                expert = expert_id
            elif s < min_dist:
                min_dist = s
                expert = expert_id

        # if num_train_samples <= 25: # for s_long
        #     all_dist = torch.tensor(all_dist)
        #     _, expert_idx =torch.sort(all_dist)
        #     for e in expert_idx:
        #         if self.model.expert2max_train_samples[e] >= 10 * num_train_samples:
        #             return "reuse", e
        
        if min_dist <= self.reuse_threshold:
            return "reuse", expert ,min_dist,all_dist,all_dist2
        # elif min_dist <= self.reuse_cell_threshold:
        #     return "reuse cell", expert
        else:
            return "new", expert,min_dist,all_dist,all_dist2
            
    def learn(self, task_id, valid_data, batch_size,device):
        """learn a task 

        """      
        if task_id == 0:
            # train
            strategy='new'
            expert_id=task_id
            min_dist=0
            all_dist=0
            all_dist2=0
        

        else:
            feat_means, feat_covs, all_feats = self.get_mean_cov_feats(
            task_id, valid_data, device=device)
            for t in range(task_id):
                self.add_mean_cov(task_id,feat_means[t],feat_covs[t])
            self.get_relatedness(task_id, all_feats)

            num_train_samples=len(valid_data)*batch_size
            strategy, expert_id,min_dist,all_dist,all_dist2 = self.strategy(task_id, num_train_samples)
        self.expert2task.append([task_id])  
        print(self.task_dist)
        print(self.task_dist2)

        return strategy, expert_id,min_dist,all_dist,all_dist2
    
    def after_learn(self, task_id, valid_data, batch_size,device):
        """learn a task 

        """ 
        feat_means, feat_covs, all_feats = self.after_get_mean_cov_feats(
            task_id, valid_data, device=device)
        self.add_mean_cov(task_id,feat_means,feat_covs)
        print(self.task_dist)
        print(self.task_dist2)
            
    
class ResNet_FE(nn.Module):
    """
	Create a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
	and get the most related model whilst training a new task in a sequence
	"""
    def __init__(self, resnet_model):
        super(ResNet_FE, self).__init__()
        self.fe_model = nn.Sequential(*list(resnet_model.children())[:-1])
        self.fe_model.eval()
        self.fe_model.requires_grad_(False)
    
    def forward(self, x):
        return self.fe_model(x)
    
def get_pretrained_feat_extractor(name):
    """get the feature extractor pretrained on ImageNet
    
    """
    if name == "resnet18":
        feat_extractor = ResNet_FE(models.resnet18(weights=True))
        # self.logger.info("Using relatedness feature extractor: ResNet18")
    else:
        raise Exception("Unknown relatedness feature extractor !!!")

    return feat_extractor

def gaussian_mean_cov(X):
    """mean and covariance of Guassian distribution
    
    Params:
        X: N x D
    """
    device = X.device
    N, D = X.shape[0], X.shape[1]
    u = torch.mean(X, dim=0)
    u_row = torch.reshape(u, (1, -1))  # 1 x D
    cov = torch.matmul(X.T, X) - N * torch.matmul(u_row.T, u_row)  # D x D
    cov = cov / (N - 1)

    cov = cov * torch.diag(torch.ones(D)).to(X.device) + (torch.diag(torch.ones(D))).to(X.device)

    return u, cov


# import torch.nn as nn
# import torchvision.models as models
# import numpy as np
# import os
# from sklearn.utils import shuffle
# import torch
# import torch.nn.functional as F

# class Appr(object):
#     """ Class implementing the TALL """
#     def __init__(self, pretrained_feat_extractor, num_task, device='cuda', args=None):

#         self.task2expert = []
#         self.expert2task = []
        
#         self.task2mean = []
#         self.task2cov = []
#         self.task_dist = torch.zeros(num_task, num_task).to(device='cuda')
#         self.feat_extractor = get_pretrained_feat_extractor(pretrained_feat_extractor).to(device='cuda')
#         self.task_relatedness_method = "mean"
#         self.reuse_threshold=0.3
#         self.reuse_cell_threshold=0.75

#     def get_mean_cov_feats(self, t, data, device):
#         """compute mean and cov for features of data extracted by the expert of task t

#         """
#         # data = deepcopy(data)  # copy for using different preprocess
        
#         # self.model.requires_grad_(False)
#         # self.model.eval()
#         # self.model.set_current_task(t)

#         class_num = 10
#         labels = torch.arange(class_num).view(-1, 1).to(device)
#         all_feats = [[] for _ in range(class_num)]

#         self.feat_extractor.eval()
#         with torch.no_grad():
#             for batch_idx, (x, y) in enumerate(data):
#                 x, y = x.to(device), y.to(device)
#                 # forward
#                 feat = self.feat_extractor(x)
#                 feat = feat.view(feat.size(0), -1)

#                 index = labels == y.view(1, -1) # CxC
#                 for i in range(class_num):
#                     all_feats[i].append(feat[index[i]])
            
            
#             all_feats_cat = [torch.cat(feats, axis=0) for feats in all_feats]
#             feat_means = []
#             feat_covs = []
#             for feat in all_feats_cat:
#                 feat_mean, feat_cov = gaussian_mean_cov(feat)
#                 feat_means.append(feat_mean)
#                 feat_covs.append(feat_cov)
#             # feat_means = [torch.mean(feat, dim=0) for feat in all_feats_cat]

#         return feat_means, feat_covs, all_feats_cat
    
#     def add_mean_cov(self, mean, cov=None):
#         self.task2mean.append(mean)
#         self.task2cov.append(cov)

#     def task_relatedness_knnkl(self, task_id, p_task_id, all_feats):
#         """
#         Params:
#             all_feat: shape C x N_c x D
#         """
#         # means and features of current data from expert of p_task_id
#         # feat_means, all_feats = means_and_feats[p_task_id]
#         # means of data of p_task_id
#         p_feat_means = self.task2mean[p_task_id] # C x D
#         feat_means = self.task2mean[task_id] # C x D
#         p_feat_means = torch.stack(p_feat_means, dim=0)

#         task_dist = 0
#         d = p_feat_means.shape[-1]
#         n = 0
#         flag = False
#         for i in range(len(feat_means)):
#             # for each current class
#             n += all_feats[i].shape[0]

#             dist_in = all_feats[i] - feat_means[i] # N_c x D
#             dist_in = torch.sqrt(torch.sum(dist_in ** 2, dim=-1)) # N_c
            
#             # N_c x C x D
#             dist_out = torch.unsqueeze(all_feats[i], dim=1) - torch.unsqueeze(p_feat_means, dim=0)
#             dist_out, _ = torch.min(torch.sqrt(torch.sum(dist_out ** 2, dim=-1)), dim=-1)  # N_c

#             dist = torch.mean(torch.log(dist_out / dist_in))

#             # if dist <= 0:
#             #     flag = True

#             task_dist += torch.maximum(dist, torch.zeros_like(dist))
        
#         # task_dist = task_dist / len(feat_means)
#         # task_dist = 1 - torch.exp(-2*task_dist)
#         task_dist = torch.minimum(1 - torch.exp(-2*task_dist), task_dist)

#         if task_dist == 0:
#             task_dist = torch.ones_like(task_dist)

#         return task_dist

#     def get_relatedness(self, task_id, feats):
#         """Compute relatedness
        
#         """
#         # the distance between task_id and task_id 
#         self.task_dist[task_id][task_id] = 0

#         for p_task_id in range(task_id):
#         # for p_task_id in range(task_id + 1):
#             # task_dist = self.task_relatedness_cos(task_id, p_task_id)
#             task_dist = self.task_relatedness_knnkl(task_id, p_task_id, feats)
#             # task_dist = self.task_relatedness_CKA(task_id, p_task_id, feats)
#             # task_dist = self.task_relatedness_gaussian_kl(task_id, p_task_id)
#             self.task_dist[task_id][p_task_id] = task_dist
#             self.task_dist[p_task_id][task_id] = task_dist
    
#     def strategy(self, task_id, num_train_samples):
#         """ Find the expert to be reused. If not found, return -1.
        
#         """
#         expert = -1
#         min_dist = None

        
#         all_dist = []
#         for expert_id, p_tasks in enumerate(self.expert2task):
#             d = self.task_dist[task_id, p_tasks]   
#             if self.task_relatedness_method == "mean":
#                 s = torch.mean(d).item()
#             elif self.task_relatedness_method == "max":
#                 s = torch.max(d).item()
#             elif self.task_relatedness_method == "min":
#                 s = torch.min(d).item()
#             else:
#                 raise Exception("Unknown reuse strategy !!!")
#             all_dist.append(s)
#             if min_dist is None:
#                 min_dist = s
#                 expert = expert_id
#             elif s < min_dist:
#                 min_dist = s
#                 expert = expert_id

#         # if num_train_samples <= 25: # for s_long
#         #     all_dist = torch.tensor(all_dist)
#         #     _, expert_idx =torch.sort(all_dist)
#         #     for e in expert_idx:
#         #         if self.model.expert2max_train_samples[e] >= 10 * num_train_samples:
#         #             return "reuse", e
        
#         if min_dist <= self.reuse_threshold:
#             return "reuse", expert ,min_dist,all_dist
#         # elif min_dist <= self.reuse_cell_threshold:
#         #     return "reuse cell", expert
#         else:
#             return "new", expert,min_dist,all_dist
            
#     def learn(self, task_id, valid_data, batch_size,device):
#         """learn a task 

#         """      
#         feat_means, feat_covs, all_feats = self.get_mean_cov_feats(
#             task_id, valid_data, device=device)
#         self.add_mean_cov(feat_means)
#         self.get_relatedness(task_id, all_feats)

#         if task_id == 0:
#             # train
#             strategy='new'
#             expert_id=task_id
#             min_dist=0
#             all_dist=0
#         else:
#             num_train_samples=len(valid_data)*batch_size
#             strategy, expert_id,min_dist,all_dist = self.strategy(task_id, num_train_samples)
#         self.expert2task.append([task_id])  

#         return strategy, expert_id,min_dist,all_dist
            
    
# class ResNet_FE(nn.Module):
#     """
# 	Create a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
# 	and get the most related model whilst training a new task in a sequence
# 	"""
#     def __init__(self, resnet_model):
#         super(ResNet_FE, self).__init__()
#         self.fe_model = nn.Sequential(*list(resnet_model.children())[:-1])
#         self.fe_model.eval()
#         self.fe_model.requires_grad_(False)
    
#     def forward(self, x):
#         return self.fe_model(x)
    
# def get_pretrained_feat_extractor(name):
#     """get the feature extractor pretrained on ImageNet
    
#     """
#     if name == "resnet18":
#         feat_extractor = ResNet_FE(models.resnet18(weights=True))
#         # self.logger.info("Using relatedness feature extractor: ResNet18")
#     else:
#         raise Exception("Unknown relatedness feature extractor !!!")

#     return feat_extractor

# def gaussian_mean_cov(X):
#     """mean and covariance of Guassian distribution
    
#     Params:
#         X: N x D
#     """
#     device = X.device
#     N, D = X.shape[0], X.shape[1]
#     u = torch.mean(X, dim=0)
#     u_row = torch.reshape(u, (1, -1))  # 1 x D
#     cov = torch.matmul(X.T, X) - N * torch.matmul(u_row.T, u_row)  # D x D
#     cov = cov / (N - 1)

#     cov = cov * torch.diag(torch.ones(D)).to(X.device) + (torch.diag(torch.ones(D))).to(X.device)

#     return u, cov
