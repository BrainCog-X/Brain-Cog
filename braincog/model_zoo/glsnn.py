import abc
from functools import partial
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseLinearModule, BaseConvModule
from braincog.utils import rand_ortho, mse
from torch import autograd


class BaseGLSNN(BaseModule):
    """
    The fully connected model of the GLSNN
    :param input_size: the shape of the input
    :param hidden_sizes: list, the number of neurons of each layer in the hidden layers
    :param ouput_size: the number of the output layers
    """

    def __init__(self, input_size=784, hidden_sizes=[800] * 3, output_size=10, opt=None):
        super().__init__(step=opt.step, encode_type=opt.encode_type)
        network_sizes = [input_size] + hidden_sizes + [output_size]
        feedforward = []
        for ind in range(len(network_sizes) - 1):
            feedforward.append(
                BaseLinearModule(in_features=network_sizes[ind], out_features=network_sizes[ind + 1], node=LIFNode))
        self.ff = nn.ModuleList(feedforward)
        feedback = []
        for ind in range(1, len(network_sizes) - 2):
            feedback.append(nn.Linear(network_sizes[-1], network_sizes[ind]))
        self.fb = nn.ModuleList(feedback)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                out_, in_ = m.weight.shape
                m.weight.data = torch.Tensor(rand_ortho((out_, in_), np.sqrt(6. / (out_ + in_))))
                m.bias.data.zero_()
        self.step = opt.step
        self.lr_target = opt.lr_target

    def forward(self, x):
        """
        process the information in the forward manner
        :param x: the input
        """
        self.reset()
        x = x.view(x.shape[0], 784)
        sumspikes = [0] * (len(self.ff) + 1)
        sumspikes[0] = x
        for ind, mod in enumerate(self.ff):
            for t in range(self.step):
                spike = mod(sumspikes[ind])
                sumspikes[ind + 1] += spike
            sumspikes[ind + 1] = sumspikes[ind + 1] / self.step
        return sumspikes

    def feedback(self, ff_value, y_label):
        """
        process information in the feedback manner and get target
        :param ff_value: the feedforward value of each layer
        :param y_label: the label of the corresponding input
        """
        fb_value = []
        cost = mse(ff_value[-1], y_label)
        P = ff_value[-1]
        h_ = ff_value[-2] - self.lr_target * torch.autograd.grad(cost, ff_value[-2], retain_graph=True)[0]
        fb_value.append(h_)
        for i in range(len(self.fb) - 1, -1, -1):
            h = ff_value[i + 1]
            h_ = h - self.fb[i](P - y_label)
            fb_value.append(h_)
        return fb_value, cost

    def set_gradient(self, x, y):
        """
        get the corresponding update of each layer
        """
        ff_value = self.forward(x)

        fb_value, cost = self.feedback(ff_value, y)

        ff_value = ff_value[1:]
        len_ff = len(self.ff)
        for idx, layer in enumerate(self.ff):
            if idx == len_ff - 1:
                layer.fc.weight.grad, layer.fc.bias.grad = autograd.grad(cost, layer.fc.parameters())
            else:
                in1 = ff_value[idx]
                in2 = fb_value[len(fb_value) - 1 - idx]
                loss_local = mse(in1, in2.detach())
                layer.fc.weight.grad, layer.fc.bias.grad = autograd.grad(loss_local, layer.fc.parameters())
        return ff_value, cost

    def forward_parameters(self):
        res = []
        for layer in self.ff:
            res += layer.parameters()
        return res

    def feedback_parameters(self):
        res = []
        for layer in self.fb:
            res += layer.parameters()
        return res


if __name__ == '__main__':
    net = BaseGLSNN()
    print(net)
