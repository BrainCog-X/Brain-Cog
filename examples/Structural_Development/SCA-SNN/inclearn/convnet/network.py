import copy
# import pdb

import torch
from torch import nn
import torch.nn.functional as F

from inclearn.tools import factory
from inclearn.convnet.imbalance import CR, All_av,BiC
from inclearn.convnet.classifier import CosineClassifier

from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.dea = cfg['dea']
        self.ft_type = cfg.get('feature_type', 'normal')
        self.at_res = cfg.get('attention_use_residual', False)
        self.div_type = cfg['div_type']
        self.reuse_oldfc = cfg['reuse_oldfc']
        self.prune = cfg.get('prune', False)
        self.reset = cfg.get('reset_se', True)
        self.torc=cfg['distillation']
        self.node =LIFNode
        self.encoder = Encoder(4, 'direct', temporal_flatten=False, layer_by_layer=False, **cfg)

        # if self.dea:
        #     print("Enable dynamical reprensetation expansion!")
        #     self.convnets = nn.ModuleList()
        #     self.convnets.append(
        #         factory.get_convnet(convnet_type,
        #                             nf=nf,
        #                             dataset=dataset,
        #                             start_class=self.start_class,
        #                             remove_last_relu=self.remove_last_relu))
        #     self.out_dim = self.convnets[0].out_dim
        #     self.c_dim=self.convnets[0].channel_dim
        # else:
        #     self.convnet = factory.get_convnet(convnet_type,
        #                                        nf=nf,
        #                                        dataset=dataset,
        #                                        remove_last_relu=self.remove_last_relu)
        #     self.out_dim = self.convnet.out_dim
        self.channel_number=[32,64,128,256]#[32,64,128,256] # [24,48,72,96] #[24,48,96,192][32,64,128,256]
        self.channel_dim=[48,96,192,384]
        self.c_number1=np.array(self.channel_number)
        if self.dea:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,c_dim=self.channel_number,cdim_cur=self.channel_number)
            )
            self.out_dim = self.channel_number[-1]
            self.out_dim_cc = self.channel_number[-1]
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
            
        self.classifier = None
        self.se = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device


        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "cr":
                self.postprocessor = CR()
            elif cfg['postprocessor']['type'].lower() == "aver":
                self.postprocessor = All_av()
            else:
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
        
        self.task_nn={}
        self.task_nn[-1]=np.array([0,0,0,0])
        self.task_nn[0]=np.array(self.channel_number)

        self.to(self.device)

    def forward(self, task,inputs,mask=None,classify=True):
        inputs = self.encoder(inputs)
        self.resetsnn()
        step = 4
        outputs = []
        if self.classifier is None:
            raise Exception("Add some classes before training.")
        
        if mask is not None:
            mat=mask.mat
            if self.torc:
                ttc=task
            else:
                ttc=-1
            for index, item in enumerate(self.convnets[ttc].parameters()):
                if len(item.size()) > 1 and item.size()[-1]!=1:
                    ww=item.data
                    item.data=ww*mat[index].cuda()

        if self.dea:
            # feature = [convnet(x) for convnet in self.convnets]
            for time in range(step):
                x_init = inputs[time]
                task_feature={}
                for t in range(task):
                    task_feature[t]={}
                    x=self.convnets[t].forward_init(x_init)
                    task_feature[t][0]=x
                    for l in range(len(self.convnets[t].layer_convnets)):
                        for lc in range(len(self.convnets[t].layer_convnets[l].conv)):
                            if lc==0 or lc==3:
                                identity = x
                            if lc==2:
                                if l>0:
                                    identity=self.convnets[t].layer_convnets[l].conv[lc](identity)
                                x=x+identity
                                task_feature[t][5*l+lc+1]=x
                            else:
                                old_tfeature=[]
                                for old_t in range(t):
                                    old_tfeature.append(task_feature[old_t][5*l+lc])
                                old_tfeature.append(x)
                                x= torch.cat(old_tfeature, 1)
                                x=self.convnets[t].layer_convnets[l].conv[lc](x)
                                if lc==4:
                                    x=x+identity
                                task_feature[t][5*l+lc+1]=x
                x=self.convnets[task].forward_init(x_init)
                for l in range(len(self.convnets[task].layer_convnets)):
                    for lc in range(len(self.convnets[task].layer_convnets[l].conv)):
                        if lc==0 or lc==3:
                            identity = x
                        if lc==2:
                            if l>0:
                                identity=self.convnets[task].layer_convnets[l].conv[lc](identity)
                            x=x+identity
                        else:
                            mid_feature=[]
                            for t in range(task):
                                mid_feature.append(task_feature[t][5*l+lc])
                            mid_feature.append(x)
                            x= torch.cat(mid_feature, 1)
                            x=self.convnets[task].layer_convnets[l].conv[lc](x)
                            if lc==4:
                                x=x+identity
                if self.torc:
                    last_feature=[]
                    for t in range(task):
                        last=len(task_feature[t])-1
                        last_feature.append(task_feature[t][last])
                    last_feature.append(x)
                    outputs.append(torch.cat(last_feature, 1))
                else:
                    x=self.convnets[task].avgpool(x)
                    x=x.view(x.size()[0],-1)
                    outputs.append(x)
            feature=sum(outputs).cuda()/ step
            last_dim =x.size(1)
            width = feature.size(1) 
            
            if self.torc:
                if self.reset:
                    se = factory.get_attention(width, self.ft_type, self.at_res).to(self.device)
                    features = se(feature)
                else:
                    features = self.se(feature)
            else:
                features=feature
   
        else:
            features = self.convnet(x)

        if self.torc:
            if classify==True:
                logits = self.convnets[-1].classifer(features)

                div_logits = self.convnets[-1].aux_classifier(features[:, -last_dim:]) if self.ntask > 1 else None
            else:
                logits=None
                div_logits=None
        else:
            if classify==True:
                logits = self.convnets[task].classifer(features)
            else:
                logits=None

            div_logits=None
        
        return {'feature': features, 'logit': logits, 'div_logit': div_logits, 'features': feature}

    def caculate_dim(self, x):
        feature = [convnet(x) for convnet in self.convnets]
        features = torch.cat(feature, 1)

        width = features.size(1)

        # se = factory.get_attention(width, self.ft_type, self.at_res).to(self.device)
        se = factory.get_attention(width, "ce", self.at_res).cuda()
        features = se(features)

        # import pdb
        # pdb.set_trace()
        return features.size(1), feature[-1].size(1)     

    @property
    def features_dim(self,ntask):
        if self.dea:
            return self.out_dim#+ntask*self.channel_number1[-1]
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes,min_dist):
        self.ntask += 1

        if self.dea:
            self._add_classes_multi_fc(n_classes,min_dist)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes,min_dist):
        self.classifier=self.convnets[-1].classifer
        if self.ntask > 1:
            if min_dist<0.1:
                min_dist=0.1
            self.channel_number1=np.array([32,64,96,128])*1*(1-math.exp(-5*min_dist)) #0.5,1,1.5,2,3,4 [24,48,72,96][16,32,48,64]
            self.channel_number1=self.channel_number1.astype(np.int64)
            self.channel_dim=self.channel_number1
            self.c_number1=self.c_number1+np.array(self.channel_number1)
            self.task_nn[self.ntask-1]=self.c_number1
            new_clf = factory.get_convnet("resnet18",c_dim=self.c_number1,cdim_cur=self.channel_number1).to(self.device)
            self.out_dim=self.out_dim+self.channel_number1[-1]
            self.out_dim_cc=self.channel_number1[-1]
            self.convnets.append(new_clf)
        
        if self.torc:
            if not self.reset:
                self.se = factory.get_attention(512*len(self.convnets), self.ft_type, self.at_res)
                self.se.to(self.device)

            if self.classifier is not None:
                weight = copy.deepcopy(self.classifier.weight.data)

            fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
            if self.classifier is not None and self.reuse_oldfc:
                fc.weight.data[:self.n_classes, :(self.out_dim - self.out_dim_cc)] = weight
            del self.classifier
            self.classifier = fc
            self.convnets[-1].classifer=self.classifier
        else:
            fc = self._gen_classifier(self.out_dim_cc, n_classes)
            del self.classifier
            self.classifier = fc
            self.convnets[-1].classifer=fc

        if self.torc:
            if self.div_type == "n+1":
                div_fc = self._gen_classifier(self.out_dim_cc, n_classes + 1)
            elif self.div_type == "1+1":
                div_fc = self._gen_classifier(self.out_dim_cc, 2)
            elif self.div_type == "n+t":
                div_fc = self._gen_classifier(self.out_dim_cc, self.ntask + n_classes)
            else:
                div_fc = self._gen_classifier(self.out_dim_cc, self.n_classes + n_classes)
            del self.aux_classifier
            self.aux_classifier = div_fc
            self.convnets[-1].aux_classifier=self.aux_classifier

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
            # classifier = CosineClassifier(in_features, n_classes).cuda()
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            # classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).cuda()
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier
    
    def resetsnn(self):
        """
        重置所有神经元的膜电位
        :return:
        """
        for mod in self.convnets.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()
