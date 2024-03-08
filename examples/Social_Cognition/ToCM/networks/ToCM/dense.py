import torch
import torch.distributions as td
import torch.nn as nn

from networks.ToCM.utils import build_model_snn


class DenseModel(nn.Module):
    def __init__(self, in_dim, out_dim, layers, hidden, activation="nn.ELU"):  # TODO  activation=nn.ELU
        super().__init__()

        self.model = build_model_snn(in_dim, out_dim, layers, hidden, activation=activation)  # no use activation

    def forward(self, features):
        return self.model(features)


class DenseBinaryModel(DenseModel):
    def __init__(self, in_dim, out_dim, layers, hidden, activation="nn.ELU"):  # 1280 7 2 256
        super().__init__(in_dim, out_dim, layers, hidden, activation=activation)

    def forward(self, features):
        # for name, p in self.model.named_parameters():
        #     print("name", name)
        #     print("p", p.shape)

        # if features.shape[1] != 40:
        #     print("features.shape[0] / 40: ", features.shape[0] / 40)
        #     features = torch.as_tensor(torch.split(features, int(features.shape[0] / 40), dim=0))
        # print("Dense features: ", features.shape)
        dist_inputs = self.model(features)  # features.shape 48 40 2 1280
        # print("dist_inputs:", dist_inputs.shape)
        return td.independent.Independent(td.Bernoulli(logits=dist_inputs), 1)

