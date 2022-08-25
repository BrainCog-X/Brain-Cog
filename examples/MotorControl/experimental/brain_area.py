import torch
import numpy as  np
import torch.nn as nn
from braincog.base.node.node import *


class MoColumnPOP(nn.Module):
    def __init__(self, 
                input_dims: int,
                pop_num: int = 16,
                embedding_dim: int = 64,
                time_window: int = 16) -> None:
        super().__init__()
        self._threshold = 1.0
        self.v_reset = 0.0
        self._time_window = time_window
        self._pop_num = pop_num
        self._node = LIFNode
        self.column_net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_dims, embedding_dim), 
                self._node(threshold=self._threshold, v_reset=self.v_reset)) 
            for _ in range(pop_num)
            ]
        )

        self.decode = nn.Linear(embedding_dim, 64)

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

    def _emb_decode(self, x):
        pop_emb_decode = []
        for net in self.column_net:
            emb = net(x)
            pop_emb_decode.append(self.decode(emb))
        return pop_emb_decode


    def forward(self, inputs):
        pop_emb_decode = self._emb_decode(inputs)
        out = sum(pop_emb_decode) / self._pop_num
        return out


class MotorCortex(nn.Module):
    def __init__(self, 
                input_dims: int,
                out_dims: int = 128,
                time_window: int = 16) -> None:
        super().__init__()
        self._threshold = 1.0
        self.v_reset = 0.0
        self._time_window = time_window
        self._node = LIFNode
        self.pfc_net = nn.Sequential(
            nn.Linear(input_dims, 512),
            self._node(threshold=self._threshold, v_reset=self.v_reset)
        )
        self.sma_net = nn.Sequential(
            nn.Linear(input_dims, 512),
            self._node(threshold=self._threshold, v_reset=self.v_reset)
        )

        self.ganglia_net = nn.Sequential(
            nn.Linear(512, 128),
            self._node(threshold=self._threshold, v_reset=self.v_reset)
        )
        self.pmc_net = nn.Sequential(
            nn.Linear(512, 512),
            self._node(threshold=self._threshold, v_reset=self.v_reset)
        )

        self.motor_net = nn.Sequential(
            nn.Linear(512+128, 128),
            self._node(threshold=self._threshold, v_reset=self.v_reset)
        )

        self.motor_emb = MoColumnPOP(input_dims=128, embedding_dim=out_dims, time_window=time_window)


    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()
        

    def _compute_motor_out(self, inputs):
        sma_out = self.sma_net(inputs)
        ganglia_out = self.ganglia_net(sma_out)
        motor_in = torch.concat([ganglia_out, sma_out], dim=-1)
        motor_out = self.motor_net(motor_in)
        # pop coding 
        return motor_out


    def forward(self, inputs):
        self.reset()
        outs = []
        for step in range(self._time_window):
            motor_out =  self._compute_motor_out(inputs)
            m_emb = self.motor_emb(motor_out)  # [Batch, 128]
            outs.append(m_emb)
        return outs
        


class Celebellum(nn.Module):
    def __init__(self,
                 input_dims: int =  512,
                 out_dims: int =  7, 
                 time_window: int = 16,
                 ) -> None:
        super().__init__()
        self._threshold = 1.0
        self.v_reset = 0.0
        self._time_window = time_window
        self._node = LIFNode
        self.gc_layer = nn.Sequential(
            nn.Linear(input_dims, 512),
            self._node(threshold=self._threshold, v_reset=self.v_reset)
        )

        self.pc_layer = nn.Sequential(
            nn.Linear(512, 512),
            self._node(threshold=self._threshold, v_reset=self.v_reset)
        )
        self.dcn_layer = nn.Sequential(
            nn.Linear(input_dims + 512, 512),
            self._node(threshold=self._threshold, v_reset=self.v_reset),
            nn.Linear(512, out_dims)
        )

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()
    def forward(self, x):
        self.reset()
        outs = []
        for step in range(self._time_window):
            gc = self.gc_layer(x[step])
            pc = self.pc_layer(gc)
            dcn_in = torch.concat([x[step], pc], dim=-1)
            dcn = self.dcn_layer(dcn_in)
            outs.append(dcn)
        cel_out = sum(outs) / self._time_window
        return cel_out



if __name__ == '__main__':
    motor = MotorCortex(input_dims=1024)
    for mod in motor.modules():
        # print('mod: ', mod)
        if hasattr(mod, 'n_reset'):
            print('mod: ', mod)
    