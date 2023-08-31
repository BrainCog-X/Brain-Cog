from functools import partial
from typing import List, Type

from braincog.model_zoo.darts.operations import *
from braincog.model_zoo.darts.genotypes import Genotype
from braincog.base.utils import drop_path
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.model_zoo.base_module import BaseModule


class MlpCell(BaseModule):
    def __init__(
        self,
        genotype: Genotype,
        C: int,
        input_dim: int,
        output_dim: int,
        encode_type: str = 'direct',
        activation_fn: Type[nn.Module] = LIFNode,
        squash_output: bool = False,
        back_connection: bool = True,
        step: int = 10,
        **kwargs
    ):
        super(MlpCell, self).__init__(
            step=step,
            encode_type=encode_type,
            layer_by_layer=True
        )
        # print(activation_fn, step)
        self.act_fun = partial(activation_fn, step=step, layer_by_layer=self.layer_by_layer, **kwargs)
        self.back_connection = back_connection
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

        self.feature = nn.Sequential(
            nn.Linear(input_dim, C),
            self.act_fun(),
        )
        if output_dim > 0:
            self.output_fn = nn.Linear(self.multiplier * C, output_dim)
        elif squash_output:
            self.output_fn = nn.Tanh()
        else:
            self.output_fn = nn.Identity()

    def _compile(self, C, op_names, indices, concat):
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
                back_begin_index = i
                break
            op = OPS_Mlp[name](C, act_fun=self.act_fun)
            self._ops += [op]

        if self.back_connection:
            for name, index in zip(op_names[back_begin_index:], indices[back_begin_index:]):
                op = OPS_Mlp[name.replace('_back', '')](
                    C, act_fun=self.act_fun)
                self._ops_back += [op]

        if self.back_connection:
            self._indices_forward = indices[:back_begin_index]
            self._indices_backward = indices[back_begin_index:]
        else:
            self._indices_backward = []
            self._indices_forward = indices
        self._steps = len(self._indices_forward) // 2

    def _forward_once(self, s0, s1, drop_prob):

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices_forward[2 * i]]
            h2 = states[self._indices_forward[2 * i + 1]]
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
                    states[self._indices_backward[i - 1]
                           ] = states[self._indices_backward[i - 1]] + s_back
            states += [s]

        outputs = []
        for i in self._concat:
            outputs.append(rearrange(states[i], '(t b) c -> t b c', t=self.step))
        outputs = torch.cat(outputs, dim=2)  # T, B, C

        return outputs

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self._forward_once(x, x, 0.)
            x = self.output_fn(x)
            x = x.mean(0)
        else:
            raise NotImplementedError

        return x


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, act_fun, back_connection):
        # print(C_prev_prev, C_prev, C, reduction)
        super(Cell, self).__init__()
        self.act_fun = act_fun
        self.back_connection = back_connection
        self.reduction = reduction
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

            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
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

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices_forward[2 * i]]
            h2 = states[self._indices_forward[2 * i + 1]]
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
                    states[self._indices_backward[i - 1]
                           ] = states[self._indices_backward[i - 1]] + s_back
            states += [s]
        outputs = torch.cat([states[i]
                            for i in self._concat], dim=1)  # N，C，H, W
        return outputs
        # return self.node(outputs)


class DCOCell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, act_fun):
        super(DCOCell, self).__init__()
        self.act_fun = act_fun

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, tos, froms = zip(*genotype.reduce)
        else:
            op_names, tos, froms = zip(*genotype.normal)
        self._compile(C, op_names, tos, froms, reduction)

    def _compile(self, C, op_names, tos, froms, reduction):
        self._ops = nn.ModuleDict()
        for name_i, to_i, from_i in zip(op_names, tos, froms):
            stride = 2 if reduction and from_i < 2 else 1
            op = OPS[name_i](C, stride, True, act_fun=self.act_fun)
            if str(to_i) in self._ops.keys():
                if str(from_i) in self._ops[str(to_i)]:
                    self._ops[str(to_i)][str(from_i)] += [op]
                else:
                    self._ops[str(to_i)][str(from_i)] = nn.ModuleList()
                    self._ops[str(to_i)][str(from_i)] += [op]
            else:
                self._ops[str(to_i)] = nn.ModuleDict()
                self._ops[str(to_i)][str(from_i)] = nn.ModuleList()
                self._ops[str(to_i)][str(from_i)] += [op]

        # TODO: Some intermediate node maybe no selected during search.
        self.multiplier = len(self._ops)

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = {}
        states['0'] = s0
        states['1'] = s1

        # get all the operations in current intermediate node
        for to_i, ops in self._ops.items():
            h = []
            for from_i, op_i in ops.items():
                # each edge may no more than one operation
                if from_i not in states:
                    # print('Exist the isolate node, which id is {}, we need ignore it!'.format(from_i))
                    continue
                h += [sum([op(states[from_i])
                          for op in op_i if from_i in states])]
            out = sum(h)
            if self.training and drop_prob > 0:
                out = drop_path(out, drop_prob)
            states[to_i] = out

        outputs = torch.cat([v for v in states.values()][2:], dim=1)
        # return outputs
        return outputs


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes, act_fun):
        """assuming inputs size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.act_fun = act_fun
        self.features = nn.Sequential(
            # nn.ReLU(inplace=True),
            self.act_fun(),
            # image size = 2 x 2
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            self.act_fun(),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            # nn.ReLU(inplace=True)
            self.act_fun()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming inputs size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


@register_model
class NetworkCIFAR(BaseModule):

    def __init__(self,
                 C,
                 num_classes,
                 layers,
                 auxiliary,
                 genotype,
                 parse_method='darts',
                 step=1,
                 node_type='ReLUNode',
                 **kwargs):
        super(NetworkCIFAR, self).__init__(
            step=step,
            num_classes=num_classes,
            **kwargs
        )
        if isinstance(node_type, str):
            self.act_fun = eval(node_type)
        else:
            self.act_fun = node_type
        self.act_fun = partial(self.act_fun, **kwargs)

        if 'back_connection' in kwargs.keys():
            self.back_connection = kwargs['back_connection']
        else:
            self.back_connection = False

        self.spike_output = kwargs['spike_output'] if 'spike_output' in kwargs else True
        self.dataset = kwargs['dataset']

        if self.layer_by_layer:
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = nn.Flatten()

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

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in self.reduce_idx:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            if parse_method == 'darts':
                cell = Cell(genotype, C_prev_prev, C_prev, C_curr,
                            reduction, reduction_prev,
                            act_fun=self.act_fun, back_connection=self.back_connection)
            else:
                cell = DCOCell(genotype, C_prev_prev, C_prev, C_curr,
                               reduction, reduction_prev, act_fun=self.act_fun)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(
                C_to_auxiliary, num_classes, act_fun=self.act_fun)
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
            for t in range(self.step):
                x = inputs[t]
                s0 = s1 = self.stem(x)
                for i, cell in enumerate(self.cells):
                    s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
                    # print(s0.shape, s1.shape)
                    # if i == 2 * self._layers // 3:
                    #     if self._auxiliary and self.training:
                    #         logits_aux = self.auxiliary_head(s1)
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


@register_model
class NetworkImageNet(BaseModule):

    def __init__(self,
                 C,
                 num_classes,
                 layers,
                 auxiliary,
                 genotype,
                 step=1,
                 node_type='ReLUNode',
                 **kwargs):
        super(NetworkImageNet, self).__init__(
            step=step,
            num_classes=num_classes,
            **kwargs)

        if isinstance(node_type, str):
            self.act_fun = eval(node_type)
        else:
            self.act_fun = node_type
        self.act_fun = partial(self.act_fun, **kwargs)

        if 'back_connection' in kwargs.keys():
            self.back_connection = kwargs['back_connection']
        else:
            self.back_connection = False

        self.spike_output = kwargs['spike_output'] if 'spike_output' in kwargs else True

        if self.layer_by_layer:
            self.flatten = nn.Flatten(start_dim=1)
        else:
            self.flatten = nn.Flatten()

        self._layers = layers
        self._auxiliary = auxiliary
        self.drop_path_prob = 0

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            # nn.ReLU(inplace=True),
            self.act_fun(),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        self.stem1 = nn.Sequential(
            # nn.ReLU(inplace=True),
            self.act_fun(),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev,
                        C_curr, reduction, reduction_prev,
                        act_fun=self.act_fun, back_connection=self.back_connection)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, inputs):
        outputs = []
        self.reset()
        for t in range(self.step):
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
            for i, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            out = self.global_pooling(s1)
            logits = self.classifier(self.flatten(out))
            outputs.append(logits)
        return sum(outputs) / len(outputs)


if __name__ == '__main__':
    from braincog.model_zoo.darts.genotypes import mlp2
    cell = MlpCell(mlp2, C=128, input_dim=17, output_dim=-1)
    x = torch.rand(4, 17)
    out = cell(x)
    print(out)
    print(out.shape)
