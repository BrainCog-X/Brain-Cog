import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from opacus import PrivacyEngine

from model import *
from braincog.base.node.node import *
import warnings
from load_data import *
from opacus.utils.batch_memory_manager import BatchMemoryManager

warnings.simplefilter("ignore")

# Precomputed characteristics of the dataset dataset
torch.cuda.manual_seed(3154)

batch_size = 512
MAX_PHYSICAL_BATCH_SIZE = 32
target_ep = 8
c = 6
epochs = 40

step = 10
delta = 1e-5
devices = 4
r = 5
device = torch.device(f'cuda:{devices}' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
disable_noise = False
data_root = "./dataset"
kwargs = {"num_workers": 1, "pin_memory": True}
dataset = 'dvs_cifar10'
# NMNIST, cifar10, dvs_cifar10, MNIST, FashionMNIST


def train(model, device, train_loader, optimizer, epoch, privacy_engine):
    criterion = nn.CrossEntropyLoss().to(device)
    losses = []
    model.train()
    correct = 0
    for _batch_idx, (data, target) in enumerate(train_loader):
        # print(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    if not disable_noise:
        epsilon = privacy_engine.get_epsilon(delta=delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
        )
        print("Accuracy: {}/{} ({:.2f}%)\n".format(
            correct,
            len(train_loader.dataset),
            100.0 * correct / len(train_loader.dataset), ))
        print(
              f"(ε = {epsilon:.2f}, δ = {delta})"
              )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    return 100.0 * correct / len(train_loader.dataset)


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def run():
    if dataset == 'dvs_cifar10':
        train_loader, test_loader, train_data, test_data = load_dvs10_data(batch_size=batch_size, step=step)
        # train_loader, test_loader, _, _ = get_dvsc10_data(batch_size=batch_size, step=step)
    elif dataset == 'NMNIST':
        train_loader, test_loader, train_data, test_data = load_nmnist_data(batch_size=batch_size, step=step)
    else:
        train_data, test_data, train_loader, test_loader = load_static_data(data_root, batch_size, dataset)

    result = []
    result_train = []
    for _ in range(r):
        if dataset == 'cifar10':
            model = cifar_convnet(
                step=step,
                encode_type='direct',
                node_type=LIFNode,
                num_classes=10,
                spike_output=False,
                layer_by_layer=True,
                act_fun=QGateGrad
            )
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1, last_epoch=-1)

        elif dataset == 'dvs_cifar10':
            model = dvs_convnet(
                step=step,
                encode_type='direct',
                node_type=LIFNode,
                num_classes=10,
                spike_output=False,
                layer_by_layer=True,
                act_fun=QGateGrad
            )
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=1, last_epoch=-1)
        elif dataset == 'NMNIST':
            model = SimpleSNN(
                channel=2,
                step=step,
                node_type=LIFNode,
                act_fun=QGateGrad,
                layer_by_layer=True,
            )
            model.to(device)

            optimizer = optim.AdamW(model.parameters(), lr=0.005)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1, last_epoch=-1)

        elif dataset == 'MNIST' or dataset == 'FashionMNIST':
            model = SimpleSNN(
                channel=1,
                step=step,
                node_type=LIFNode,
                act_fun=QGateGrad,
                layer_by_layer=True,
            )
            model.to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.005)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1, last_epoch=-1)
        if not disable_noise:
            privacy_engine = PrivacyEngine()
            model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                max_grad_norm=c,
                epochs=epochs,
                target_delta=delta,
                target_epsilon=target_ep
            )
            with BatchMemoryManager(
                    data_loader=data_loader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=optimizer
            ) as memory_safe_data_loader:
                # if 1:
                for epoch in range(1, epochs + 1):
                    result_train.append(train(model, device, memory_safe_data_loader, optimizer, epoch, privacy_engine))
                    result.append(test(model, device, test_loader))
                    scheduler.step()
        else:
            privacy_engine = PrivacyEngine()
            model, optimizer, data_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                max_grad_norm=c,
                noise_multiplier=0.0,
            )
            with BatchMemoryManager(
                    data_loader=data_loader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=optimizer
            ) as memory_safe_data_loader:
                for epoch in range(1, epochs + 1):
                    train(model, device, memory_safe_data_loader, optimizer, epoch, privacy_engine)
                    result.append(test(model, device, test_loader))
                    scheduler.step()
    result = np.array(result).reshape((r, -1))
    result_train = np.array(result_train).reshape((r, -1))
    best_acc = np.mean(np.max(result, axis=1))
    print(best_acc)
    np.save(file=f'./{dataset}/MP_test.npy', arr=result)
    np.save(file=f'./{dataset}/MP_train.npy', arr=result_train)

if __name__ == "__main__":
    run()
