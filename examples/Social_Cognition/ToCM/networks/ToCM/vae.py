import torch.nn as nn
import torch.nn.functional as F

from networks.ToCM.utils import build_model_snn


class Decoder(nn.Module):

    def __init__(self, embed, hidden, out_dim, layers=2):
        super().__init__()
        self.fc1 = build_model_snn(embed, hidden, layers, hidden, activation='nn.ReLU')  # activation=nn.ReLU
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        return self.fc2(x), x


class Encoder(nn.Module):

    def __init__(self, in_dim, hidden, embed, layers=2):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden)
        self.encoder = build_model_snn(hidden, embed, layers, hidden, activation='nn.ReLU')   # activation=nn.ReLU

    def forward(self, x):
        embed = F.relu(self.fc1(x))
        return self.encoder(F.relu(embed))
