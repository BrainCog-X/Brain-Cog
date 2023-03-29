import sys
sys.path.append('../../../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import os
import math
from tqdm import tqdm
import numpy as np
from braincog.datasets.datasets import get_mnist_data
from braincog.base.node import LIFNode
from braincog.utils import setup_seed


setup_seed(1111)
os.environ['CUDA_VISIBLE_DEVICES'] = "4"


class myLIFnode(LIFNode):
    def __init__(self, threshold=0.5, tau=2., *args, **kwargs):
        super().__init__(threshold, tau, *args, **kwargs)

    def integral(self, inputs):
        # self.mem = self.mem + (inputs - self.mem) / self.tau
        self.mem = self.mem / self.tau + inputs


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u.permute(0,2,3,4,1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return u


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
        self.bias = nn.Parameter(torch.randn(out_channels, 1))

        self.W.data.normal_(0, math.sqrt(3.0 / (in_channels * out_channels)))
        self.bias.data.normal_(0, math.sqrt(3.0 / (in_channels * out_channels)))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x) + self.bias
        return u_hat


class DigitCaps2(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6):
        super(DigitCaps2, self).__init__()

        self.num_routes = num_routes
        self.num_capsules = num_capsules
        self.b_ij = Variable(torch.ones(1, self.num_routes, self.num_capsules, 1)/1152)
        self.b_ij = self.b_ij.to(device)

    def forward(self, u_hat):
        c_ij = torch.cat([self.b_ij] * batch_size, dim=0).unsqueeze(4)
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
        return s_j.squeeze(1)

    def init_bij(self):
        self.b_ij = Variable(torch.ones(1, self.num_routes, self.num_capsules, 1)/1152)
        self.b_ij = self.b_ij.to(device)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        classes = torch.sqrt((x ** 2).sum(2))
        # classes = self.linear(x)
        return classes


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.digit_capsules2 = DigitCaps2()
        self.decoder = Decoder()

        self.conv_node = myLIFnode(tau=5)
        self.primary_node = myLIFnode(tau=5)
        self.digit_node = myLIFnode(tau=5)
        self.digit2_node = myLIFnode(tau=5)

    def forward(self, data, time_window=5, train=True):
        self.init()
        out_mem = 0.
        self.digit_capsules2.init_bij()
        self.trace_u = torch.zeros(batch_size, 1152, 10, 16, 1, device=device)

        for step in range(time_window):
            x = data

            x = self.conv_node(self.conv_layer(x))
            x = self.primary_node(self.primary_capsules(x))
            x1 = self.digit_node(self.digit_capsules(x))
            x = self.digit_capsules2(x1)
            out_mem += x.squeeze(3)
            y = self.digit2_node(x)

            if train:
                with torch.no_grad():
                    self.digit_capsules2.b_ij = torch.clamp(self.digit_capsules2.b_ij, -0.05, 1)
                    self.trace_u *= torch.exp(-1 / torch.tensor(1.5))
                    self.trace_u.masked_fill_(x1 != 0, 1)
                    self.digit_capsules2.b_ij += 0.0008 * torch.matmul(
                        self.trace_u.transpose(3, 4) - 0.1,
                        torch.stack([y] * 1152, dim=1)).squeeze(4).mean(dim=0, keepdim=True)

        output = out_mem / time_window
        output = self.decoder(output)
        return output


    def init(self):
        self.conv_node.n_reset()
        self.primary_node.n_reset()
        self.digit_node.n_reset()
        self.digit2_node.n_reset()


def evaluate(test_iter, net, device):
    net.eval()

    test_loss, test_acc, n_test = 0, 0.0, 0
    for batch_id, (data, target) in tqdm(enumerate(test_iter)):
        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)

        classes = net(data)

        test_acc += sum(np.argmax(classes.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        n_test += data.shape[0]
    net.train()

    return test_acc / n_test


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 100

    train_loader, test_loader, _, _ = get_mnist_data(batch_size)
    capsule_net = CapsNet().to(device)
    optimizer = Adam(capsule_net.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()

    n_epochs = 50
    best, losses = 0, []

    for epoch in range(n_epochs):
        if epoch in [15, 25, 45]:
            optimizer.param_groups[0]['lr'] *= 0.3

        capsule_net.train()
        train_loss, correct, n = 0, 0, 0
        loss_rec = []
        for batch_id, (data, target) in enumerate(train_loader):
            target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            classes = capsule_net(data)
            loss = loss_fn(classes, target)
            loss.backward()
            loss_rec.append(loss.item())
            optimizer.step()

            train_loss += loss.item()
            correct += sum(np.argmax(classes.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
            n += data.shape[0]

            if batch_id % 100 == 0:
                print("Epoch: {}, Batch: {}, train accuracy: {:.6f}, loss: {:.6f}".format(
                    epoch, batch_id + 1,
                    sum(np.argmax(classes.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size),
                    loss.item()))
        losses.append(np.mean(np.array(loss_rec)))

        print("Epoch: [{}/{}],  train accuracy: {:.6f}, loss: {:.6f}".format(
            epoch, n_epochs,
            correct / float(n),
            train_loss / len(train_loader)))


        capsule_net.eval()
        test_acc = evaluate(test_loader, capsule_net, device=device)
        print("test accuracy: {:.6f}".format(test_acc))

        if test_acc > best:
            best = test_acc
            # torch.save(capsule_net, './checkpoints/spikingcaps_mnist.pkl')

