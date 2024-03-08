import os
import copy
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from braincog.datasets.datasets import get_mnist_data
from braincog.model_zoo.qsnn import Net
from braincog.datasets.gen_input_signal import lambda_max


LOG_DIR = os.path.expanduser('./results.txt')

LEARNING_RATE = 0.01
# learning dacay
DECAY_STEPS = 1.0
DECAY_RATE = 0.9
# adam
BETA1 = 0.9
BETA2 = 0.999
EPSIOLN = 1e-8

EPOCHS = 20
PRINT_PERIOD = 10000
TEST_SIZE = 10000

TEST_THETA = [0, 1 / 16, 2 / 16, 3 / 16, 4 / 16, 5 / 16, 6 / 16, 7 / 16, 8 / 16]
# TEST_THETA = [0]
NOISE_RATES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


train_loader, test_loader, _, _ = get_mnist_data(batch_size=1, skip_norm=True)

NET_SIZE = [28 * 28, 500, 10]


def int2onehot(label, classes, factor):
    label_one_hot = F.one_hot(label, classes)
    label_one_hot = label_one_hot * (8 + factor) - 8
    return label_one_hot


def train(net, epochs, lr):
    with open(LOG_DIR, 'a+') as f:
        for epoch in range(epochs):
            lr_decay = lr * DECAY_RATE ** (epoch / DECAY_STEPS)
            for x, y in tqdm.tqdm(train_loader):
                label = int2onehot(y, 10, 8)
                x = x.flatten().numpy()
                label = label.cuda()
                with torch.no_grad():
                    net.routine(x, None, image_ori=None, image_ori_delta=None, shift=False,
                                label=label, test=False, noise=False, noise_rate=None)
                    net.update_weight(lr_decay, epoch + 1, (BETA1, BETA2), EPSIOLN)
            with torch.no_grad():
                for fac in TEST_THETA:
                    acc_reve = 0
                    for x_test, y_test in test_loader:
                        image = x_test.flatten().numpy()
                        image_shift = image * np.cos(fac * np.pi) + (lambda_max - image) * np.sin(fac * np.pi)
                        image_delta = copy.copy(image)
                        delta_idx = image_delta < (lambda_max - 0.001)
                        image_delta[delta_idx] += 0.001
                        image_delta_shift = image_delta * np.cos(fac * np.pi) + (lambda_max - image_delta) * np.sin(fac * np.pi)
                        pred = net.predict(image_shift, image_delta_shift, image, image_delta, shift=True, noise=False, noise_rate=None)
                        if pred == int(y_test):
                            acc_reve += 1
                    acc_reve = acc_reve / TEST_SIZE
                    print('Test epoch {epoch}: Shift {theta:0.3f} pi: accuracy {acc}.'.format(
                        epoch=epoch, theta=fac, acc=acc_reve))
                    print('Test epoch {epoch}: Shift {theta:0.3f} pi: accuracy {acc}'.format(
                        epoch=epoch, theta=fac, acc=acc_reve), file=f)
                    print()
                    print(file=f)


if __name__ == '__main__':
    net = Net(NET_SIZE).cuda()
    train(net, EPOCHS, LEARNING_RATE)
