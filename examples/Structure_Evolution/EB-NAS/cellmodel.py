import os
from functools import partial
from typing import List, Type

from operations import *
from motifs import *
from utils import drop_path
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.model_zoo.base_module import BaseModule
from torchvision import transforms

EVO=True
class EvoCell2(nn.Module):
    def __init__(self,motif, C_prev_prev, C_prev, C, reduction, reduction_prev, act_fun):
        # print(C_prev_prev, C_prev, C, reduction)
        super(EvoCell2, self).__init__()
        self.act_fun = act_fun
        self.reduction = reduction
        self.motif=motif
        self.back_connection=False
        if reduction:
            self.fun = FactorizedReduce(
                C_prev, C * 3, act_fun=act_fun
            )
            self.multiplier = 3
        else:
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(
                    C_prev_prev, C, act_fun=act_fun)
            else:
                self.preprocess0 = ReLUConvBN(
                    C_prev_prev, C, 1, 1, 0, act_fun=act_fun)
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, act_fun=act_fun)

            op_names, indices = zip(*motif.normal)
            concat = motif.normal_concat
            self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        # self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._ops_back = nn.ModuleList()
        back_begin_index = 0
        for i, (name, index) in enumerate(zip(op_names, indices)):
            # print(name, index)
            if '_back' in name:
                self.back_connection=True
                back_begin_index = i
                break
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, act_fun=self.act_fun)
            self._ops += [op]

        if self.back_connection:
            for name, index in zip(op_names[back_begin_index:], indices[back_begin_index:]):
                op = OPS[name.replace('_back', '')](
                    C, 1, True, act_fun=self.act_fun)
                self._ops_back += [op]

        if self.back_connection:
            self._indices_forward = indices[:back_begin_index]
            self._indices_backward = indices[back_begin_index:]
        else:
            self._indices_backward = []
            self._indices_forward = indices
        self._steps = len(self._indices_forward) // 2

    def forward(self, s0, s1, drop_prob):
        if self.reduction:
            return self.fun(s1)
        # print('s0',s0.shape)
        s0 = self.preprocess0(s0)
        # print(s0.shape)
        # print('s1',s1.shape)
        s1 = self.preprocess1(s1)
        # print(s1.shape)

        states = [s0, s1]
        for i in range(self._steps):
            i1=self._indices_forward[2 * i]
            i2=self._indices_forward[2 * i + 1]
            h1 = states[i1]
            h2 = states[i2]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            
            if self.back_connection:
                if i != 0:
                    s_back = self._ops_back[i - 1](s)
                    states[self._indices_backward[i - 1]] = states[self._indices_backward[i - 1]] + s_back
            states += [s]
        
            
        
        outputs = torch.cat([states[i]
                            for i in self._concat], dim=1)  # N，C，H, W
        return outputs
        # return self.node(outputs)


class EvoCell3(nn.Module):
    def __init__(self,motif, C_prev_prev_prev, C_prev_prev, C_prev, C, reduction, reduction_prev, reduction_prev_prev, act_fun):
        # print(C_prev_prev_prev,C_prev_prev, C_prev, C, reduction,reduction_prev, reduction_prev_prev)

        super(EvoCell3, self).__init__()
        self.act_fun = act_fun
        self.reduction = reduction
        self.motif=motif
        self.back_connection=False
        if reduction:
            self.fun = FactorizedReduce(C_prev, C * 3, act_fun=act_fun)
            self.multiplier = 3
        else:

            if reduction_prev:
                self.preprocess1 = FactorizedReduce(C_prev_prev, C, act_fun=act_fun)
            else:
                self.preprocess1 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, act_fun=act_fun)


            if int(reduction_prev_prev)+int(reduction_prev)==1:
                self.preprocess0 = FactorizedReduce(C_prev_prev_prev, C, act_fun=act_fun)
            elif int(reduction_prev_prev)+int(reduction_prev)==2:
                self.preprocess0 = F0(C_prev_prev_prev, C, act_fun=act_fun)
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev_prev, C, 1, 1, 0, act_fun=act_fun)


            self.preprocess2 = ReLUConvBN(C_prev, C, 1, 1, 0, act_fun=act_fun)


            op_names, indices = zip(*motif.normal)
            concat = motif.normal_concat
            self._compile(C, op_names, indices, concat, reduction)
    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        # self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._ops_back = nn.ModuleList()
        back_begin_index = 0
        for i, (name, index) in enumerate(zip(op_names, indices)):
            # print(name, index)
            if '_back' in name:
                self.back_connection=True
                back_begin_index = i
                break
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, act_fun=self.act_fun)
            self._ops += [op]

        if self.back_connection:
            for name, index in zip(op_names[back_begin_index:], indices[back_begin_index:]):
                op = OPS[name.replace('_back', '')](
                    C, 1, True, act_fun=self.act_fun)
                self._ops_back += [op]

        if self.back_connection:
            self._indices_forward = indices[:back_begin_index]
            self._indices_backward = indices[back_begin_index:]
        else:
            self._indices_backward = []
            self._indices_forward = indices
        self._steps = len(self._indices_forward) // 3

    def forward(self, s0, s1, s2, drop_prob):
        if self.reduction:
            return self.fun(s2)

        s0 = self.preprocess0(s0)

        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)

        states = [s0, s1, s2]

        for i in range(self._steps):
            i1=self._indices_forward[3 * i]
            i2=self._indices_forward[3 * i + 1]
            i3=self._indices_forward[3 * i + 2]

            h1 = states[i1]
            h2 = states[i2]
            h3 = states[i3]

            op1 = self._ops[3 * i]
            op2 = self._ops[3 * i + 1]
            op3 = self._ops[3 * i + 2]
            h1 = op1(h1)
            h2 = op2(h2)
            h3 = op3(h3)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)                
                if not isinstance(op3, Identity):
                    h3 = drop_path(h3, drop_prob)
            s = h1 + h2 + h3
            
            if self.back_connection:
                if i != 0:
                    s_back = self._ops_back[i - 1](s)
                    states[self._indices_backward[i - 1]] = states[self._indices_backward[i - 1]] + s_back
            states += [s]
        
            
        
        outputs = torch.cat([states[i] for i in self._concat], dim=1)  # N，C，H, W
        return outputs
        # return self.node(outputs)

class EvoCell4(nn.Module):
    def __init__(self,motif, C_prev_prev_prev_prev,C_prev_prev_prev, C_prev_prev, C_prev, C, reduction, reduction_prev, reduction_prev_prev,reduction_prev_prev_prev, act_fun):
        # print(C_prev_prev_prev_prev,C_prev_prev_prev,C_prev_prev, C_prev, C, reduction,reduction_prev, reduction_prev_prev,reduction_prev_prev_prev)

        super(EvoCell4, self).__init__()
        self.act_fun = act_fun
        self.reduction = reduction
        self.motif=motif
        self.back_connection=False
        if reduction:
            self.fun = FactorizedReduce(C_prev, C * 3, act_fun=act_fun)
            self.multiplier = 3
        else:

            if reduction_prev:
                self.preprocess2 = FactorizedReduce(C_prev_prev, C, act_fun=act_fun)
            else:
                self.preprocess2 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, act_fun=act_fun)

            if int(reduction_prev_prev)+int(reduction_prev)==1:
                self.preprocess1 = FactorizedReduce(C_prev_prev_prev, C, act_fun=act_fun)
            elif int(reduction_prev_prev)+int(reduction_prev)==2:
                self.preprocess1 = F0(C_prev_prev_prev, C, act_fun=act_fun)
            else:
                self.preprocess1 = ReLUConvBN(C_prev_prev_prev, C, 1, 1, 0, act_fun=act_fun)
            
            if int(reduction_prev_prev_prev)+int(reduction_prev_prev)+int(reduction_prev)==1:
                self.preprocess0 = FactorizedReduce(C_prev_prev_prev_prev, C, act_fun=act_fun)
            elif int(reduction_prev_prev_prev)+int(reduction_prev_prev)+int(reduction_prev)==2:
                self.preprocess0 = F0(C_prev_prev_prev_prev, C, act_fun=act_fun)            
            elif int(reduction_prev_prev_prev)+int(reduction_prev_prev)+int(reduction_prev)==3:
                self.preprocess0 = F1(C_prev_prev_prev_prev, C, act_fun=act_fun)
            else:
                self.preprocess0 = ReLUConvBN(C_prev_prev_prev_prev, C, 1, 1, 0, act_fun=act_fun)



            self.preprocess3 = ReLUConvBN(C_prev, C, 1, 1, 0, act_fun=act_fun)


            op_names, indices = zip(*motif.normal)
            # print(self.preprocess0)
            # print(self.preprocess1)
            # print(self.preprocess2)
            # print(self.preprocess3)
            concat = motif.normal_concat
            self._compile(C, op_names, indices, concat, reduction)
    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        # self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._ops_back = nn.ModuleList()
        back_begin_index = 0
        for i, (name, index) in enumerate(zip(op_names, indices)):
            # print(name, index)
            if '_back' in name:
                self.back_connection=True
                back_begin_index = i
                break
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, act_fun=self.act_fun)
            self._ops += [op]

        if self.back_connection:
            for name, index in zip(op_names[back_begin_index:], indices[back_begin_index:]):
                op = OPS[name.replace('_back', '')](
                    C, 1, True, act_fun=self.act_fun)
                self._ops_back += [op]

        if self.back_connection:
            self._indices_forward = indices[:back_begin_index]
            self._indices_backward = indices[back_begin_index:]
        else:
            self._indices_backward = []
            self._indices_forward = indices
        self._steps = len(self._indices_forward) // 4

    def forward(self, s0, s1, s2, s3, drop_prob):
        if self.reduction:
            return self.fun(s3)

        s0 = self.preprocess0(s0)
        s3 = self.preprocess3(s3)
        s1 = self.preprocess1(s1)
        s2 = self.preprocess2(s2)

        # if s1.shape[1]!=s3.shape[1]:
        #     s1 = nn.Conv2d(s1.shape[1], s3.shape[1], 3, stride=2, padding=1, bias=False)

        states = [s0, s1, s2,s3]

        for i in range(self._steps):
            i1=self._indices_forward[4 * i]
            i2=self._indices_forward[4 * i + 1]
            i3=self._indices_forward[4 * i + 2]
            i4=self._indices_forward[4 * i + 3]

            h1 = states[i1]
            h2 = states[i2]
            h3 = states[i3]
            h4 = states[i4]

            op1 = self._ops[4 * i]
            op2 = self._ops[4 * i + 1]
            op3 = self._ops[4 * i + 2]
            op4 = self._ops[4 * i + 3]
            h1 = op1(h1)
            h2 = op2(h2)
            h3 = op3(h3)
            h4 = op4(h4)

            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)                
                if not isinstance(op3, Identity):
                    h3 = drop_path(h3, drop_prob)                
                if not isinstance(op4, Identity):
                    h4= drop_path(h4, drop_prob)
            s = h1 + h2 + h3 + h4
            
            if self.back_connection:
                if i != 0:
                    s_back = self._ops_back[i - 1](s)
                    states[self._indices_backward[i - 1]] = states[self._indices_backward[i - 1]] + s_back
            states += [s]
        
            
        
        outputs = torch.cat([states[i] for i in self._concat], dim=1)  # N，C，H, W
        return outputs
        # return self.node(outputs)



@register_model

class NetworkCIFAR(BaseModule):

    def __init__(self,
                 C,
                 num_classes,
                 layers,
                 auxiliary,
                 motif,
                 cell_type,
                 parse_method='darts',
                 step=5,
                 node_type='ReLUNode',
                 **kwargs):
        super(NetworkCIFAR, self).__init__(
            step=step,
            num_classes=num_classes,
            **kwargs
        )
        self.node_type=node_type
        if isinstance(node_type, str):
            self.act_fun = eval(node_type)
        else:
            self.act_fun = node_type
        self.act_fun = partial(self.act_fun, **kwargs)
        
        self.spike_output = kwargs['spike_output'] if 'spike_output' in kwargs else True
        self.dataset = kwargs['dataset']

        if self.layer_by_layer:
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = nn.Flatten()

        self._layers = layers
        self.cell_type = cell_type
        self._auxiliary = auxiliary

        self.drop_path_prob = 0

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        if self.dataset == 'dvsg' or self.dataset == 'dvsc10' or self.dataset == 'NCALTECH101':
            self.stem = nn.Sequential(
                nn.Conv2d(2 * self.init_channel_mul, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            # self.reduce_idx = [
            #     layers // 4,
            #     layers // 2,
            #     3 * layers // 4
            # ]
            self.reduce_idx = [1, 3, 5, 7]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3 * self.init_channel_mul, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            self.reduce_idx = [layers // 4,
                               layers // 2,
                               3 * layers // 4]
        C_prev_prev_prev = C_curr
        C_prev_prev_prev_prev = C_curr

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        reduction_prev_prev = False
        reduction_prev_prev_prev = False


        for i in range(layers):
            if i in self.reduce_idx:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            if cell_type==2:
                # print(C_prev_prev, C_prev, C_curr)

                cell = EvoCell2(motif[i], C_prev_prev, C_prev, C_curr,reduction, reduction_prev,act_fun=self.act_fun)
                self.cells += [cell]
                C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            if cell_type==3:
                cell = EvoCell3(motif[i], C_prev_prev_prev, C_prev_prev, C_prev, C_curr,reduction, reduction_prev,reduction_prev_prev,act_fun=self.act_fun)  
                self.cells += [cell]
                C_prev_prev_prev = C_prev_prev
                reduction_prev_prev = reduction_prev

                C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            if cell_type==4:
                cell = EvoCell4(motif[i], C_prev_prev_prev_prev,C_prev_prev_prev, C_prev_prev, C_prev, C_curr,reduction, reduction_prev,reduction_prev_prev,reduction_prev_prev_prev,act_fun=self.act_fun)  
                self.cells += [cell]
                C_prev_prev_prev_prev = C_prev_prev_prev
                C_prev_prev_prev = C_prev_prev
                reduction_prev_prev_prev = reduction_prev_prev
                reduction_prev_prev = reduction_prev

                C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr


            reduction_prev = reduction


        self.global_pooling = nn.Sequential(
            self.act_fun(), nn.AdaptiveAvgPool2d(1))

        if self.spike_output:
            self.classifier = nn.Sequential(
                nn.Linear(C_prev, 10 * num_classes),
                self.act_fun())
            self.vote = VotingLayer(10)
        else:
            self.classifier = nn.Linear(C_prev, num_classes)
            self.vote = nn.Identity()

        # self.classifier = nn.Linear(C_prev, num_classes)
        # self.vote = nn.Identity()

    def forward(self, inputs):
        logits_aux = None
        inputs = self.encoder(inputs)
        if not self.layer_by_layer:
            outputs = []
            output_aux = []
            self.reset()

            if self.cell_type==2:

                for t in range(self.step):
                    x = inputs[t]
                    s0 = s1 = self.stem(x)
                    for i, cell in enumerate(self.cells):
                        s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
                    out = self.global_pooling(s1)
                    out = self.classifier(self.flatten(out))
                    logits = self.vote(out)
                    outputs.append(logits)
                    output_aux.append(logits_aux)
                return sum(outputs) / len(outputs)

            if self.cell_type==3:
                for t in range(self.step):
                    x = inputs[t]
                    s0 = s1 = s2= self.stem(x)
                    for i, cell in enumerate(self.cells):
                        s0, s1, s2 = s1, s2, cell(s0, s1, s2, self.drop_path_prob)

                    out = self.global_pooling(s2)
                    out = self.classifier(self.flatten(out))
                    logits = self.vote(out)
                    outputs.append(logits)
                    output_aux.append(logits_aux)
                return sum(outputs) / len(outputs)

            if self.cell_type==4:
                for t in range(self.step):
                    x = inputs[t]
                    s0 = s1 = s2= s3=self.stem(x)
                    for i, cell in enumerate(self.cells):
                        s0, s1, s2,s3= s1, s2, s3,cell(s0, s1, s2,s3 ,self.drop_path_prob)

                    out = self.global_pooling(s3)
                    out = self.classifier(self.flatten(out))
                    logits = self.vote(out)
                    outputs.append(logits)
                    output_aux.append(logits_aux)
                return sum(outputs) / len(outputs)
                
            



            # logits_aux if logits_aux is None else (sum(output_aux) / len(output_aux))
        else:
            s0 = s1 = self.stem(inputs)
            for i, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
                if i == 2 * self._layers // 3:
                    if self._auxiliary and self.training:
                        logits_aux = self.auxiliary_head(s1)
            out = self.global_pooling(s1)
            out = self.classifier(self.flatten(out))
            out = rearrange(out, '(t b) c -> t b c', t=self.step).mean(0)
            logits = self.vote(out)
            return logits


@register_model

class NetworkCIFAR_(BaseModule):

    def __init__(self,
                 C,
                 num_classes,
                 layers,
                 glob,
                 auxiliary,
                 motif,
                 parse_method='darts',
                 step=5,
                 node_type='ReLUNode',
                 **kwargs):
        super(NetworkCIFAR_, self).__init__(
            step=step,
            num_classes=num_classes,
            **kwargs
        )
        self.node_type=node_type
        if isinstance(node_type, str):
            self.act_fun = eval(node_type)
        else:
            self.act_fun = node_type
        self.act_fun = partial(self.act_fun, **kwargs)
        
        self.spike_output = kwargs['spike_output'] if 'spike_output' in kwargs else True
        self.dataset = kwargs['dataset']

        if self.layer_by_layer:
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = nn.Flatten()
        self.glob = glob
        self._layers = layers
        self._auxiliary = auxiliary

        self.drop_path_prob = 0

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        if self.dataset == 'dvsg' or self.dataset == 'dvsc10' or self.dataset == 'NCALTECH101':
            self.stem = nn.Sequential(
                nn.Conv2d(2 * self.init_channel_mul, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            # self.reduce_idx = [
            #     layers // 4,
            #     layers // 2,
            #     3 * layers // 4
            # ]
            self.reduce_idx = [1, 3, 5, 7]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3 * self.init_channel_mul, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            self.reduce_idx = [layers // 4,
                               layers // 2,
                               3 * layers // 4]
        C_prev_prev_prev = C_curr
        C_prev_prev_prev_prev = C_curr

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        reduction_prev_prev = False
        reduction_prev_prev_prev = False


        for i in range(layers):
            if i in self.reduce_idx:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = EvoCell2(motif[i], C_prev_prev, C_prev, C_curr,reduction, reduction_prev,act_fun=self.act_fun)
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            reduction_prev = reduction


        self.global_pooling = nn.Sequential(
            self.act_fun(), nn.AdaptiveAvgPool2d(1))

        if self.spike_output:
            self.classifier = nn.Sequential(
                nn.Linear(C_prev, 10 * num_classes),
                self.act_fun())
            self.vote = VotingLayer(10)
        else:
            self.classifier = nn.Linear(C_prev, num_classes)
            self.vote = nn.Identity()

        # self.classifier = nn.Linear(C_prev, num_classes)
        # self.vote = nn.Identity()

    def forward(self, inputs):
        logits_aux = None
        inputs = self.encoder(inputs)
        if not self.layer_by_layer:
            outputs = []
            output_aux = []
            self.reset()

            zzz=[]
            kkk=[]

            for t in range(self.step):

                x = inputs[t]
                s0 = s1 = self.stem(x)
                # print(s1.shape)
                for i, cell in enumerate(self.cells):
                    
                    if t>0 and i%5==4:
                        qw = np.where(self.glob[:,int(i//5)]==1)
                        if qw[0].shape[0]!=0:

                            for m in qw:
                                if zzz[m[0]].shape[-1]>s1.shape[-1]:
                                    ks=zzz[m[0]].shape[-1] - (s1.shape[-1]-1)*2+2
                                    if ks<0:
                                        ks=zzz[m[0]].shape[-1] - (s1.shape[-1]-1)+2
                                        bb=nn.Conv2d(zzz[m[0]].shape[1], s1.shape[1], kernel_size=ks, stride=1,padding=1, bias=False).to(zzz[m[0]].device)
                                    else:
                                        bb=nn.Conv2d(zzz[m[0]].shape[1], s1.shape[1], kernel_size=ks, stride=2,padding=1, bias=False).to(zzz[m[0]].device)
                                    aa=bb(zzz[m[0]])
                                    s1=aa+s1
                                elif zzz[m[0]].shape[-1]<s1.shape[-1]:

                                    aa=nn.functional.interpolate(zzz[m[0]],[s1.shape[-1], s1.shape[-1]], mode='bilinear', align_corners=False)
                                    bb=nn.Conv2d(zzz[m[0]].shape[1], s1.shape[1],kernel_size=1).to(zzz[m[0]].device)
                                    aa=bb(aa)
                                    s1=aa+s1




                    s0,s1=s1,cell(s0, s1, self.drop_path_prob)
                    if i%5==4:
                        if t==0:
                            zzz.append(s1)
                            kkk.append(s0)
                        else:
                            zzz[int(i//5)] = s1
                            kkk[int(i//5)] = s0


                out = self.global_pooling(s1)
                out = self.classifier(self.flatten(out))
                logits = self.vote(out)
                outputs.append(logits)
                output_aux.append(logits_aux)
            return sum(outputs) / len(outputs)


            # logits_aux if logits_aux is None else (sum(output_aux) / len(output_aux))
        else:
            s0 = s1 = self.stem(inputs)
            for i, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
                if i == 2 * self._layers // 3:
                    if self._auxiliary and self.training:
                        logits_aux = self.auxiliary_head(s1)
            out = self.global_pooling(s1)
            out = self.classifier(self.flatten(out))
            out = rearrange(out, '(t b) c -> t b c', t=self.step).mean(0)
            logits = self.vote(out)
            return logits


def occumpy_mem(cuda_device):
    total, used = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")[int(cuda_device)].split(',')
    # total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 1)
    block_mem = int((max_mem - used)*0.85)
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x

if __name__ == '__main__':
    torch.cuda.set_device('cuda:3')
    # occumpy_mem(str(3))

    x = torch.rand(128, 3, 32, 32)
    glob = np.array([[0,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,0,0]])
    # glob = np.array([[0,1],[1,0]])
    glob = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
    glob = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    glob = np.array([[0,1,0],[1,0,0],[1,1,1]])

    motifs=[mm2,mm3,mm4,mm5,mm1,mm5,mm3,mm4,mm2,mm1,mm2,mm3,mm4,mm5,mm1]
    # motifs=[m1,m2,m3,m1,m5,m4,m1,m2,m3,m1,m5,m4,m5,m4,m1]##3
    # motifs=[t2,t3,t4,t5,t1,t5,t3,t4,t2,t1,t2,t3,t4,t5,t1,t5,t3,t4,t2,t1]
    # motifs=[t2,t3,t4,t5,t1]

    # motifs=[subnet,subnet,subnet]

    net=NetworkCIFAR_(C=12,num_classes=10,motif=motifs,layers=len(motifs),auxiliary=True,dataset='cifar10',glob=glob)
    # net=NetworkCIFAR(C=12,num_classes=10,motif=motifs,layers=len(motifs),auxiliary=True,dataset='cifar10',cell_type=2)

    net=net.cuda()
    layers=int(len(motifs)/5)
    out=net(x.to('cuda:3'))
    print(out.shape)
