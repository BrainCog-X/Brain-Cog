from functools import partial
from braincog.model_zoo.darts.operations import *
from torch.autograd import Variable
from braincog.model_zoo.darts.genotypes import PRIMITIVES
from braincog.model_zoo.darts.genotypes import Genotype
from . import parse

from braincog.base.connection.layer import VotingLayer
from braincog.base.node.node import *
from braincog.model_zoo.base_module import BaseModule
from . import forward_edge_num
from . import edge_num


def calc_weight(x):
    tmp0 = torch.split(x[0], edge_num, dim=0)
    tmp1 = torch.split(x[1], edge_num, dim=0)
    res = []
    for i in range(len(edge_num)):
        res.append(
            torch.softmax(tmp0[i].view(-1), dim=-1).view(tmp0[i].shape)
            + torch.softmax(tmp1[i].view(-1), dim=-1).view(tmp1[i].shape)
        )
    return torch.cat(res, dim=0)


def calc_loss(x):
    tmp0 = torch.split(x[0], edge_num, dim=0)
    tmp1 = torch.split(x[1], edge_num, dim=0)
    res = []
    for i in range(len(edge_num)):
        res.append(
            torch.softmax(tmp0[i].view(-1), dim=-1).view(tmp0[i].shape)
            - torch.softmax(tmp1[i].view(-1), dim=-1).view(tmp1[i].shape)
        )
    return torch.cat(res, dim=0)


class darts_fun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):  # feature map / arch weight
        output = inputs * weights
        ctx.save_for_backward(inputs, weights)
        return output

    @staticmethod
    def backward(ctx, grad_output):  # error signal
        grad_inputs, grad_weights = None, None

        inputs, weights = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output * weights
        if ctx.needs_input_grad[1]:
            if torch.min(inputs) < -1e-12 and torch.max(inputs) > 1e-12:
                inputs = torch.abs(inputs) / 2.
            else:
                inputs = torch.abs(inputs)
            grad_weights = -inputs.mean()

        return grad_inputs, grad_weights


class MixedOp(nn.Module):
    def __init__(self, C, stride, act_fun):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False, act_fun)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

        self.multiply = darts_fun.apply

    def forward(self, x, weights):
        feature_map = []
        for i, op in enumerate(self._ops):
            res = op(x)
            feature_map.append(res)
        return sum(self.multiply(mp, w) for w, mp in zip(weights, feature_map))


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, act_fun, back_connection):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.back_connection = back_connection
        if reduction:
            self.fun = FactorizedReduce(
                C_prev, C * multiplier, affine=True, act_fun=act_fun, positive=1
            )
        else:
            if reduction_prev:
                self.preprocess0 = FactorizedReduce(
                    C_prev_prev, C, affine=False, act_fun=act_fun, positive=1)
            else:
                self.preprocess0 = ReLUConvBN(
                    C_prev_prev, C, 1, 1, 0, affine=False, act_fun=act_fun, positive=1)
            self.preprocess1 = ReLUConvBN(
                C_prev, C, 1, 1, 0, affine=False, act_fun=act_fun, positive=1)
            self._steps = steps
            self._multiplier = multiplier

            self._ops = nn.ModuleList()

            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(C, stride, act_fun)
                    self._ops.append(op)

            if self.back_connection:
                self._ops_back = nn.ModuleList()
                for i in range(self._steps):
                    for j in range(i):
                        op = MixedOp(C, 1, act_fun)
                        self._ops_back.append(op)

    def forward(self, s0, s1, weights):
        if self.reduction:
            return self.fun(s1)

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        offset_back = 0

        weights_forward = weights[:forward_edge_num]
        weights_backward = weights[forward_edge_num:]
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights_forward[offset + j])
                    for j, h in enumerate(states))
            offset += len(states)

            if self.back_connection:
                for j in range(2, len(states)):
                    # print(j, len(states), offset_back, len(self._ops_back))
                    states[j] = states[j] + \
                        self._ops_back[offset_back](
                            s, weights_backward[offset_back])
                    offset_back += 1

            states.append(s)

        outputs = torch.cat(states[-self._multiplier:], dim=1)
        return outputs


class Network(BaseModule):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3,
                 parse_method='bio_darts', op_threshold=None, step=1, node_type='ReLUNode', **kwargs):

        super().__init__(
            step=step,
            encode_type='direct',
            **kwargs
        )

        self.act_fun = eval(node_type)
        self.act_fun = partial(self.act_fun, **kwargs)

        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier
        self.parse_method = parse_method
        self.op_threshold = op_threshold
        self.fire_rate_per_step = [0.] * self.step
        self.forward_step = 0
        self.record_fire_rate = False
        if 'back_connection' in kwargs.keys():
            self.back_connection = kwargs['back_connection']
        else:
            self.back_connection = False
        self.dataset = kwargs['dataset']
        self.spike_output = kwargs['spike_output'] if 'spike_output' in kwargs else True

        C_curr = stem_multiplier * C

        if self.dataset == 'dvsg' or self.dataset == 'dvsc10' or self.dataset == 'NCALTECH101':
            self.stem = nn.Sequential(
                nn.Conv2d(2 * self.init_channel_mul, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            self.reduce_idx = [layers // 3,
                               2 * layers // 3]
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3 * self.init_channel_mul, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr),
            )
            self.reduce_idx = [1, 3, 5]

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in self.reduce_idx:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, self.act_fun,
                        self.back_connection)
            reduction_prev = reduction
            self.cells += [cell]

            C_prev_prev, C_prev = C_prev, multiplier * C_curr
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
        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes,
                            self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, inputs):
        inputs = self.encoder(inputs)

        self.reset()
        if not self.training:
            self.fire_rate.clear()

        outputs = []
        for t in range(self.step):
            x = inputs[t]
            s0 = s1 = self.stem(x)
            for i, cell in enumerate(self.cells):
                if not cell.reduction:
                    weights = calc_weight(self.alphas_normal)
                    s0, s1 = s1, cell(s0, s1, weights)
                else:
                    s0, s1 = s1, cell(s0, s1, None)
            out = self.global_pooling(s1)
            out = self.classifier(out.view(out.size(0), -1))
            logits = self.vote(out)
            outputs.append(logits)
        # print(self.get_fire_rate_avg(), self.fire_rate_per_step, len(self.fire_rate_per_step))
        if self.record_fire_rate:
            self.forward_step += 1
        return sum(outputs) / len(outputs)

    def reset_fire_rate_record(self):
        self.fire_rate_per_step = [0.] * self.step
        self.forward_step = 0

    def get_fire_per_step(self):
        return [x / self.forward_step for x in self.fire_rate_per_step]

    def _loss(self, input1, target1, input2):
        logits = self(input1)
        return self._criterion(logits, target1, input2)
    # def _loss(self, input1, target1):
    #     logits = self(input1)
    #     return self._criterion(logits, target1)

    def _initialize_alphas(self):
        # k = 2 + 3 + 4 + 5 = 14
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        if self.back_connection:
            k += sum(1 for i in range(self._steps) for n in range(i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(
            0.5 * torch.randn(2, k, num_ops).cuda(), requires_grad=True)

        # init the history
        self.alphas_normal_history = {}
        mm = 0
        last_id = 1
        node_id = 0
        for i in range(k):
            for j in range(num_ops):
                self.alphas_normal_history['edge: {}, op: {}'.format(
                    (node_id, mm), PRIMITIVES[j])] = []
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1

    def arch_parameters(self):
        return [self.alphas_normal]

    def genotype(self):

        # alphas_normal
        gene_normal = parse(calc_weight(self.alphas_normal).data.cpu().numpy(),
                            PRIMITIVES, self.op_threshold, self.parse_method,
                            self._steps, reduction=False, back_connection=self.back_connection)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
        )
        return genotype

    def states(self):
        return {
            'alphas_normal': self.alphas_normal,
            'alphas_normal_history': self.alphas_normal_history,
            'criterion': self._criterion
        }

    def restore(self, states):
        self.alphas_normal = states['alphas_normal']
        self.alphas_normal_history = states['alphas_normal_history']

    def update_history(self):

        mm = 0
        last_id = 1
        node_id = 0
        weights1 = calc_weight(self.alphas_normal).data.cpu().numpy()

        k, num_ops = weights1.shape
        for i in range(k):
            for j in range(num_ops):
                self.alphas_normal_history['edge: {}, op: {}'.format((node_id, mm), PRIMITIVES[j])].append(
                    float(weights1[i][j]))
            if mm == last_id:
                mm = 0
                last_id += 1
                node_id += 1
            else:
                mm += 1
