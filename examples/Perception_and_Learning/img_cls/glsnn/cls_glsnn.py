import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from braincog.model_zoo.glsnn import BaseGLSNN
import argparse
import time
import os

import json

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser("description = GLSNN.py")
parser.add_argument('-seed', type=int, default=2122)
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-lr_target', type=float, default=0.4)
parser.add_argument('-lr_forward', type=float, default=0.001)
parser.add_argument('-step', type=int, default=10)
parser.add_argument('-encode_type', type=str, default='direct')
parser.add_argument('--dataset', type=str, default='MNIST')
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

test_scores = []
train_scores = []

save_path = './' + 'GLSNN' + '_' + opt.dataset + '_' + str(opt.seed)
if not os.path.exists(save_path):
    os.mkdir(save_path)
if opt.dataset == 'MNIST':
    train_dataset = datasets.MNIST(root='./data/datasets/mnist/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/datasets/mnist/', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
elif opt.dataset == 'Fashion-MNIST':
    train_dataset = datasets.FashionMNIST(root='./data/fashion/', train=True, transform=transforms.ToTensor(),
                                          download=True)
    test_dataset = datasets.FashionMNIST(root='./data/fashion/', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)

snn = BaseGLSNN(input_size=784, hidden_sizes=[800] * 3, output_size=10, opt=opt)
snn.to(device)
optimizer = torch.optim.Adam(snn.forward_parameters(), lr=opt.lr_forward)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


def train(epoch):
    snn.train()

    start_time = time.time()
    total_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels_ = torch.zeros(opt.batch_size, 10).scatter_(1, labels.view(-1, 1), 1).to(device)
        labels = labels.to(device)

        outputs, loss = snn.set_gradient(images, labels_)
        optimizer.step()
        total_loss += loss.item()
        pred = outputs[-1].max(1)[1]
        total += labels.size(0)
        correct += (pred.cpu() == labels.cpu()).sum()
        if (i + 1) % (60000 // (opt.batch_size * 6)) == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Time: %.2f' % (
                epoch + 1, opt.epoch, i + 1, 60000 // opt.batch_size, total_loss,
                time.time() - start_time))
            start_time = time.time()
            total_loss = 0
    acc = 100.0 * correct.item() / total
    train_scores.append(acc)


def eval(epoch):
    snn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = snn(images)
            pred = outputs[-1].max(1)[1]
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()
    acc = 100.0 * correct.item() / total
    print('Test correct: %d Accuracy: %.2f%%' % (correct, acc))
    test_scores.append(acc)
    if acc >= max(test_scores):
        save_file = str(epoch) + '.pt'
        torch.save(snn, os.path.join(save_path, save_file))
    return max(test_scores)


def main():
    for epoch in range(opt.epoch):
        train(epoch)
        best_acc = eval(epoch)
        scheduler.step()
        print('Best Accuracy: %.2f%%' % (best_acc))


if __name__ == '__main__':
    main()
    filename = "train.json"
    filename = os.path.join(save_path, filename)
    with open(filename, "w") as f:
        json.dump(train_scores, f)
    filename = "test.json"
    filename = os.path.join(save_path, filename)
    with open(filename, "w") as f:
        json.dump(test_scores, f)
