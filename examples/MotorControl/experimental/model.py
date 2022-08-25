import torch
import numpy as  np
import torch.nn as nn
from brain_area import Celebellum, MotorCortex


class Motion(nn.Module):
    def __init__(self, in_dims: int, out_dims: int=17, time_window: int=8, emb_size: int = 128) -> None:
        super().__init__()
        self._time_window = time_window
        self.in_emb = nn.Linear(in_dims, emb_size)
        self.motor_cotex = MotorCortex(input_dims=emb_size, out_dims=64,  time_window=self._time_window)
        self.cele = Celebellum(input_dims=64, out_dims=out_dims, time_window=self._time_window)
        # self.opti = torch.optim.Adam(net.parameters(), lr=0.001)

    def forward(self, x):
        in_emb = self.in_emb(x)
        motor_out = self.motor_cotex(in_emb)
        out = self.cele(motor_out)
        return out

    def learn(self):
        pass

