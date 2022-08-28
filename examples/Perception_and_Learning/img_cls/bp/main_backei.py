import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
from braincog.model_zoo.backeinet import *
import argparse
import os
import json



parser = argparse.ArgumentParser("description = train.py")
parser.add_argument('-seed', type=int, default=4150)
parser.add_argument('-epoch', type=int, default=200)
parser.add_argument('-batch_size', type=int, default=100)
parser.add_argument('-learning_rate', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='fashion')
parser.add_argument('--simulation_len', type=int, default=20)
parser.add_argument('--Back', action='store_true', default=False)
parser.add_argument('--EI', action='store_true', default=False)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--encode-type', type=str, default='direct')
opt = parser.parse_args()
torch.cuda.set_device('cuda:%d' % opt.device)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

test_scores = []
train_scores = []
save_path = opt.dataset + '_' + str(opt.seed) + '_' + opt.encode_type
if opt.Back:
    save_path += '_Back'
if opt.EI:
    save_path += '_EI'

if not os.path.exists(save_path):
    os.mkdir(save_path)
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
if opt.dataset == 'mnist':
    train_dataset = datasets.MNIST(root='./data/mnist/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/mnist/', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
elif opt.dataset == 'fashion':
    train_dataset = datasets.FashionMNIST(root='./data/fashion/', train=True, transform=transforms.ToTensor(),
                                          download=True)
    test_dataset = datasets.FashionMNIST(root='./data/fashion/', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)

elif opt.dataset == 'cifar10':
    train_dataset = datasets.CIFAR10(root='./data/cifar10/', train=True, transform=transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), normalize]),
                                     download=True)
    test_dataset = datasets.CIFAR10(root='./data/cifar10/', train=False,
                                    transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False)
if opt.dataset == 'cifar10':
    snn = CIFARNet(step=opt.simulation_len, if_back=opt.Back, if_ei=opt.EI,  encode_type=opt.encode_type)
else:
    snn = MNISTNet(step=opt.simulation_len, if_back=opt.Back, if_ei=opt.EI, data=opt.dataset, encode_type=opt.encode_type)
snn = snn.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


def train(epoch):
    snn.train()

    start_time = time.time()
    total_loss = 0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.cuda()
        outputs = snn(images)
        labels_ = torch.zeros(opt.batch_size, 10).scatter_(1, labels.view(-1, 1), 1).cuda()
        loss = criterion(outputs, labels_)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = outputs.max(1)[1]
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
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
            images = images.cuda()
            outputs = snn(images)

            pred = outputs.max(1)[1]
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
