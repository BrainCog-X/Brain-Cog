import torch
from torch import nn
from braincog.base.encoder import encoder
from braincog.base.node import LIFNode


class CNNQnet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNQnet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.l = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_dim)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.l(x)
        return x


class SNNQnet(nn.Module):
    def __init__(self, input_dim, output_dim,
                 step=4, node=LIFNode, encode_type='direct'):
        super(SNNQnet, self).__init__()
        self.step = step
        self.encoder = encoder.Encoder(step=step, encode_type=encode_type)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1, padding_mode='replicate'),
            node(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            node(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            node(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.l = nn.Sequential(
            nn.Linear(in_features=64, out_features=128),
            node(),
            nn.Linear(in_features=128, out_features=output_dim)
        )

    def forward(self, input):
        inputs = self.encoder(input)
        outputs = []
        self.reset()
        for t in range(self.step):
            x = inputs[t]
            x = self.cnn(x)
            x = self.l(x)
            outputs.append(x)
        return sum(outputs) / len(outputs)

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()
