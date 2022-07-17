import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm 
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import cv2
import numpy as np
from copy import deepcopy
import os, time, math,random
from  BrainCog.base.node.node import * 
from BrainCog.base.connection .layer import *
from BrainCog.base.strategy.LateralInhibition import *
from sklearn.metrics import confusion_matrix

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
  
torch.cuda.manual_seed(seed) #GPU随机种子确定 

torch.backends.cudnn.benchmark = False #模型卷积层预先优化关闭
torch.backends.cudnn.deterministic = True #确定为默认卷积算法

random.seed(seed) 

os.environ["PYTHONHASHSEED"] = str(seed) 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dev = "cuda"
device = torch.device(dev) if torch.cuda.is_available() else 'cpu'
torch.set_printoptions(precision=4, sci_mode=False)


# ===========================================================================================================

convoff = 0.3


# avgscale = 5


class STDPConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding,groups,
                 tau_decay=torch.exp(-1.0 / torch.tensor(100.0)), offset=convoff, static=True, inh=6.5, avgscale=5):
        super().__init__()
        self.tau_decay = tau_decay
        self.offset = offset
        self.static = static
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups,
                              bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.mem = self.spike = self.refrac_count = None
        self.normweight()
        self.inh = inh
        self.avgscale = avgscale
        self.onespike=True
        self.node=LIFSTDPNode(act_fun=STDPGrad,tau=tau_decay,mem_detach=True)
        self.WTA=WTALayer( )
        self.lateralinh=LateralInhibition(self.node,self.inh,mode="threshold")
        
    def mem_update(self, x, onespike=True):  # b,c,h,w

        x=self.node( x)
               
        if x.max() > 0:
            x=self.WTA(x)
        
            self.lateralinh(x)

        self.spike= x 
        return self.spike

    def forward(self, x, T=None, onespike=True):

        if not self.static:
            batch, T, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)

        x = self.conv(  x)

        n = self.getthresh(x)
        self.node.threshold.data = n

        x=x.clamp(min=0)
        x = n / (1 + torch.exp(-(x - 4 * n / 10) * (8 / n)))

        if not self.static:
            x = x.reshape(batch, T, c, h, w)
            xsum = None
            for i in range(T):
                tmp = self.mem_update(x[:, i], onespike).unsqueeze(1)
                if xsum is not None:
                    xsum = torch.cat([xsum, tmp], 1)
                else:
                    xsum = tmp
        else:
            xsum = 0
            for i in range(T):
                xsum += self.mem_update(x, onespike)

        return xsum

    def reset(self):
        #self.mem = self.spike = self.refrac_count = None
        self.node.n_reset()
    def normgrad(self, force=False):
        if force:
            min = self.conv.weight.grad.data.min(1, True)[0].min(2, True)[0].min(3, True)[0]
            max = self.conv.weight.grad.data.min(1, True)[0].max(2, True)[0].max(3, True)[0]
            self.conv.weight.grad.data -= min
            tmp = self.offset * max
        else:
            tmp = self.offset * self.spike.mean(0, True).mean(2, True).mean(3, True).permute(1, 0, 2, 3)
        self.conv.weight.grad.data -= tmp
        self.conv.weight.grad.data = -self.conv.weight.grad.data

    def normweight(self, clip=False):
        if clip:
            self.conv.weight.data = torch. \
                clamp(self.conv.weight.data, min=-3, max=1.0)
        else:
            c, i, w, h = self.conv.weight.data.shape

            avg=self.conv.weight.data.mean(1, True).mean(2, True).mean(3, True)
            self.conv.weight.data -=avg

            tmp = self.conv.weight.data.reshape(c, 1, -1, 1)

            self.conv.weight.data /= tmp.std(2, unbiased=False, keepdim=True)


    def getthresh(self, scale):

        tmp2= scale.max(0, True)[0].max(2, True)[0].max(3, True)[0]+0.0001

        return tmp2


class STDPLinear(nn.Module):
    def __init__(self, in_planes, out_planes,
                 tau_decay=0.99, offset=0.05, static=True,inh=10):
        super().__init__()
        self.tau_decay = tau_decay
        self.offset = offset
        self.static = static
        self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.mem = self.spike = self.refrac_count = None
        # torch.nn.init.xavier_uniform_(self.linear.weight, gain=1)
        self.normweight(False)
        self.threshold = torch.ones(out_planes, device=device) *20
        
        self.inh=inh
        self.node=LIFSTDPNode(act_fun=STDPGrad,tau=tau_decay  ,mem_detach=True)
        self.WTA=WTALayer( )
        self.lateralinh=LateralInhibition(self.node,self.inh,mode="max")
        self.init=False 
    def mem_update(self, x, onespike=True):  # b,c,h,w
        if not self.init: 
            self.node.threshold.data= (x.max(0)[0].detach()*3).to(device) 
            self.init=True

        xori=x
        x=self.node( x)
        if x.max() > 0:
            x=self.WTA(x)
        
            self.lateralinh(x,xori)

        self.spike=x
        return self.spike

    def forward(self, x, T, onespike=True):

        if not self.static:
            batch, T, w = x.shape
            x = x.reshape(-1, w)
        x = x.detach()


        
        x = self.linear(x)
        self.x=x.detach()

        if not self.static:
            x = x.reshape(batch, T, w)
            xsum = None
            for i in range(T):
                tmp = self.mem_update(x[:, i], onespike).unsqueeze(1)
                if xsum is not None:
                    xsum = torch.cat([xsum, tmp], 1)
                else:
                    xsum = tmp
        else:
            xsum = 0
            for i in range(T):
                xsum += self.mem_update(x, onespike)
        #print(xsum.mean())
        return xsum

    def reset(self):

        self.node.n_reset()
    def normgrad(self, force=False):
        if force:

            pass
        else:
            tmp = self.offset * self.spike.mean(0, True).permute(1, 0)


        self.linear.weight.grad.data = -self.linear.weight.grad.data


    def normweight(self, clip=False):

        if clip:
            self.linear.weight.data = torch. \
                clamp(self.linear.weight.data, min=0, max=1.0)
        else:
            self.linear.weight.data = torch. \
                clamp(self.linear.weight.data, min=0, max=1.0)
            sumweight = self.linear.weight.data.sum(1, True)
            sumweight += (~(sumweight.bool())).float()
            # self.linear.weight.data *= 11.76  / sumweight
            self.linear.weight.data /= self.linear.weight.data.max(1, True)[0] / 0.1

    def getthresh(self, scale):
        tmp = self.linear.weight.clamp(min=0) * scale
        tmp2 = tmp.sum(1, True).reshape(1, -1)
        return tmp2

    def updatethresh(self, plus=0.05):

        self.node.threshold += (plus*self.x * self.spike.detach()).sum(0)
        tmp=self.node.threshold.max()-350
        if tmp>0:
            self.node.threshold-=tmp

class STDPFlatten(nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=start_dim, end_dim=end_dim)

    def forward(self, x, T):  # [batch,T,c,w,h]
 
        return self.flatten(x)


class STDPMaxPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, static=True):
        super().__init__()
        self.static = static
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x, T):  # [batch,T,c,w,h]

        if not self.static:
            batch, T, c, h, w = x.shape
            x = x.reshape(-1, c, h, w)
        x = self.pool(x)
        if not self.static:
            x = x.reshape(batch, T, c, h, w)

        return x


alpha = 1.0


class Normliaze(nn.Module):
    def __init__(self, static=True):
        super().__init__()
        self.static = static

    def forward(self, x, T):  # [batch,T,c,w,h]
        # print(x.shape)
        x /= x.max(1, True)[0].max(2, True)[0].max(3, True)[0]
        # x/=x.mean()/0.13

        return x


class voting(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.label = torch.zeros(shape) - 1
        self.assignments=0
    def assign_labels(self, spikes, labels, rates=None, n_labels=10, alpha=alpha):
        # 根据最后一层的spikes 以及 label 对于最后一层的神经元赋予不同的label
        # spikes 是 batch * time * in_size
        # print(spikes.size())
        n_neurons = spikes.size(2)
        if rates is None:
            rates = torch.zeros(n_neurons, n_labels, device=device)
        self.n_labels = n_labels
        spikes = spikes.cpu().sum(1).to(device)

        for i in range(n_labels):
            n_labeled = torch.sum(labels == i).float()
            # 就是说上一次assign label计算的rates 拿过来滑动平均一下   #这里似乎可以改
            if n_labeled > 0:
                indices = torch.nonzero(labels == i).view(-1)
                tmp = torch.sum(spikes[indices], 0) / n_labeled  # 平均脉冲数
                rates[:, i] = alpha * rates[:, i] + tmp

        # 此时的rates是 in_size * n_label, 对应哪个label的rates最高 该神经元就对应着该label
        self.assignments = torch.max(rates, 1)[1]
        return self.assignments, rates

    def get_label(self, spikes):
        # 根据最后一层的spike 计算得到label
        n_samples = spikes.size(0)
        spikes = spikes.cpu().sum(1).to(device)
        rates = torch.zeros(n_samples, self.n_labels, device=device)

        for i in range(self.n_labels):
            n_assigns = torch.sum(self.assignments == i).float()  # 共有多少个该类别节点
            if n_assigns > 0:
                indices = torch.nonzero(self.assignments == i).view(-1)  # 找到该类别节点位置
                rates[:, i] = torch.sum(spikes[:, indices], 1) / n_assigns  # 该类别平均所有该类别节点发放脉冲数

        return torch.sort(rates, dim=1, descending=True)[1][:, 0]

inh=25
inh2=1.625
channel=12
neuron=6400
class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        self.conv = nn.ModuleList([
            STDPConv(1, channel, 3, 1, 1,1, static=True, inh=1.625, avgscale=5 ),
            STDPMaxPool(2, 2, 0, static=True),
            Normliaze(),
            #STDPConv(12, 48, 3, 1, 1,1, static=True, inh=inh2, avgscale=10 ),
            #STDPMaxPool(2, 2, 0, static=True),
            #Normliaze(),

            STDPFlatten(start_dim=1),
            STDPLinear(196*channel, neuron, static=True,inh=inh)

        ])

        self.voting = voting(10)

    def forward(self, x, inlayer, outlayer, T, onespike=True):  # [b,t,w,h]

        for i in range(inlayer, outlayer + 1):
            x = self.conv[i](x, T)
        return x

    def normgrad(self, layer, force=False):
        self.conv[layer].normgrad(force)

    def normweight(self, layer, clip=False):
        self.conv[layer].normweight(clip)

    def updatethresh(self, layer, plus=0.05):
        self.conv[layer].updatethresh(plus)

    def reset(self, layer):
        if isinstance(layer, list):
            for i in layer:
                self.conv[i].reset()
        else:
            self.conv[layer].reset()


def plot_confusion_matrix(cm, classes, normalize=True, title='Test Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure()
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        plt.text(i, i, format(cm[i, i], fmt), horizontalalignment="center",
                 color="white" if cm[i, i] > thresh else "black")
    plt.tight_layout()
    #plt.savefig('confusestpf2'+str(channel)+"_n"+str(neuron)+".pdf")
    #plt.show()
if __name__ == '__main__':
    print(23)
    batch_size = 1024
    T = 100
    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor()])
    # mnist_train = datasets.CIFAR10(root='/data/datasets/CIFAR10/', train=True, download=False, transform=transform )
    # mnist_test = datasets.CIFAR10(root='/data/datasets/CIFAR10/', train=False, download=False, transform=transform )
    #mnist_train = datasets.FashionMNIST(root='/data/dyt//', train=True, download=True, transform=transform )
    #mnist_test = datasets.FashionMNIST(root='/data/dyt/', train=False, download=False, transform=transform )
    mnist_train = datasets.MNIST(root='./', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./', train=False, download=False, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    model = Conv_Net().to(device)
    convlist = [index for index, i in enumerate(model.conv) if isinstance(i, (STDPConv, STDPLinear))]
    print(convlist)
    #cap = torch.ones([100000, 1000, 30], device=device)
    
    for layer in range(len(convlist) - 1):
        optimizer = torch.optim.SGD(list(model.parameters())[layer:layer + 1], lr=0.1)
        for epoch in range(3):
            for step, (x, y) in enumerate(tqdm(train_iter)):
                x = x.to(device)
                y = y.to(device)

                spikes = model(x, 0, convlist[layer], T)

                optimizer.zero_grad()
                spikes.sum().backward(torch.tensor(1/  (spikes.shape[0] * spikes.shape[2] * spikes.shape[3])))
                # spikes.sum().backward(  )
                model.conv[convlist[layer]].spike = spikes.detach()
                model.normgrad(convlist[layer], force=True)
                optimizer.step()
                model.normweight(convlist[layer], clip=False)
                # print(model.conv[convlist[layer]].conv.weight.data )
                model.reset(convlist)

            print("layer", layer, "epoch", epoch, 'Done')
        #model.conv[convlist[layer]].onespike=False
    # ===========================================================================================================
    # linear
    #model.conv[convlist[-2]].onespike=True 
    cap = None
    batch_size = 1024
    T = 200
    layer = len(convlist) - 1
    plus = 0.002
    lr = 0.0001
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)
    optimizer = torch.optim.SGD(list(model.parameters())[layer:], lr=lr)

    rates = None
    best = 0
    accrecord=[]
    for epoch in range(1000):
        spikefull = None
        labelfull = None
        for step, (x, y) in enumerate(tqdm(train_iter)):
            x = x.to(device)
            y = y.to(device)

            spiketime = 0

            spikes = model(x, 0, convlist[layer], T)
            # print(spikes.mean())
            optimizer.zero_grad()
            spikes.sum().backward()
            model.conv[convlist[layer]].spike = spikes.detach()
            model.normgrad(convlist[layer], force=False)
            optimizer.step()
            model.updatethresh(convlist[layer], plus=plus)
            model.normweight(convlist[layer], clip=False)

            spikes = spikes.reshape(spikes.shape[0], 1, -1).detach()
            if spikefull is None:
                spikefull = spikes
                labelfull = y
            else:
                spikefull = torch.cat([spikefull, spikes], 0)
                labelfull = torch.cat([labelfull, y], 0)

            model.reset(convlist)

        _, rates = model.voting.assign_labels(spikefull, labelfull, rates)
        rates = rates.detach() * 0.5
        result = model.voting.get_label(spikefull)
        acc = (result == labelfull).float().mean()

        print(epoch, acc, 'channel', channel, "n", neuron)
        print(model.conv[-1].node.threshold.max(),model.conv[-1].node.threshold.mean(),model.conv[-1].node.threshold.min())
        
        # model.conv[-1].threshold*=0.98
        spikefull = None
        labelfull = None
        result = None
        for step2, (x, y) in enumerate(test_iter):
            x = x.to(device)
            y = y.to(device)

            spiketime = 0
            spikes = model(x, 0, convlist[layer], T)

            spikes = spikes.reshape(spikes.shape[0], 1, -1).detach()

            with torch.no_grad():
                if spikefull is None:
                    spikefull = spikes
                    labelfull = y

                else:
                    spikefull = torch.cat([spikefull, spikes], 0)
                    labelfull = torch.cat([labelfull, y], 0)

            model.reset(convlist)

        result = model.voting.get_label(spikefull)
        acc = (result == labelfull).float().mean()
        if best < acc: 
            best = acc 
            torch.save( model, "modelftstp28_350_c"+str(channel)+"_n"+str(neuron)+"_p"+str(acc)+".pth")
            classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            
            cm = confusion_matrix(labelfull.cpu(), result.cpu())
            plot_confusion_matrix(cm, classes)
        print("test", acc, "best", best)
        accrecord.append(acc)
        #torch.save(accrecord,"accfstp28_350_c"+str(channel)+"_n"+str(neuron)+".pth")