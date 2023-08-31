import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from numpy.linalg import eigvals
from braincog.model_zoo.darts.model_search import calc_weight, calc_loss


def normalize(x):
    mu = np.average(x)
    sigma = np.std(x)
    return (x - mu) / sigma


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    def __init__(self, model, args):

        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.arch_parameters(),
                                           lr=args.arch_learning_rate,
                                           betas=(args.arch_lr_gamma, 0.999),
                                           weight_decay=args.arch_weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.arch_parameters(), lr=args.arch_learning_rate)
        self.hessian = None
        self.grads = None

    def step(self, input_valid, target_valid):
        self.optimizer.zero_grad()
        aux_input = torch.cat([calc_loss(self.model.alphas_normal)], dim=0)
        loss, loss1, loss2 = self.model._loss(
            input_valid, target_valid, aux_input)
        # loss = self.model._loss(input_valid, target_valid)
        loss.backward()
        self.optimizer.step()
        return loss1, loss2

    def compute_Hw(self, input_valid, target_valid):
        self.zero_grads(self.model.parameters())
        self.zero_grads(self.model.arch_parameters())
        aux_input = torch.cat(
            [F.softmax(self.model.alphas_normal, dim=-1)], dim=0)
        loss = self.model._loss(input_valid, target_valid, aux_input)
        self.hessian = self._hessian(loss, self.model.arch_parameters())
        return self.hessian

    def zero_grads(self, parameters):
        for p in parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def compute_eigenvalues(self):
        self.compute_Hw()
        return eigvals(self.hessian.cpu().data.numpy())

    def _hessian(self, outputs, inputs, out=None, allow_unused=False):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        else:
            inputs = list(inputs)

        n = sum(p.numel() for p in inputs)
        if out is None:
            out = torch.tensor(torch.zeros(n, n)).type_as(outputs)

        ai = 0
        for i, inp in enumerate(inputs):
            [grad] = torch.autograd.grad(outputs, inp, create_graph=True,
                                         allow_unused=allow_unused)
            grad = grad.contiguous().view(-1) + self.weight_decay * inp.view(-1)
            for j in range(inp.numel()):
                if grad[j].requires_grad:
                    row = self.gradient(
                        grad[j], inputs[i:], retain_graph=True)[j:]
                else:
                    n = sum(x.numel() for x in inputs[i:]) - j
                    row = Variable(torch.zeros(n)).type_as(grad[j])

                out.data[ai, ai:].add_(row.clone().type_as(out).data)
                if ai + 1 < n:
                    out.data[ai + 1:,
                             ai].add_(row.clone().type_as(out).data[1:])
                del row
                ai += 1
            del grad
        return out
