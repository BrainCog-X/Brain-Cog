from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from braincog.base.node.node import LIFNode
from ..utils.normalization import PopNorm


class SpikingDQN(nn.Module):
    """Reference: Human-level control through deep reinforcement learning.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        c: int,
        h: int,
        w: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        time_window: int = 16,
        features_only: bool = False,
    ) -> None:
        super().__init__()
        self._node = LIFNode
        # self._node = ReLUNode
        self.features_only = features_only
        self.device = device
        # self._threshold = 0.5
        self._threshold = 1.0
        self.v_reset = 0.0
        # self._decay = 0.2
        self._decay = 0.5
        self._time_window = time_window

        init_layer = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=1)
        self.p_count = 0

        self.net = nn.Sequential(
           # nn.utils.weight_norm(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            # self._node(threshold=self._threshold, decay=self._decay),
            PopNorm([32, 20, 20], threshold=self._threshold, v_reset=self.v_reset),

            self._node(threshold=self._threshold, v_reset=self.v_reset),
            # nn.utils.weight_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            # self._node(threshold=self._threshold, decay=self._decay),
            PopNorm([64, 9, 9], threshold=self._threshold, v_reset=self.v_reset),
            self._node(threshold=self._threshold, v_reset=self.v_reset),
            # nn.utils.weight_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            PopNorm([64, 7, 7], threshold=self._threshold, v_reset=self.v_reset),
            # self._node(threshold=self._threshold, decay=self._decay),
            self._node(threshold=self._threshold, v_reset=self.v_reset),
            nn.Flatten()
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, c, h, w)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, nn.Linear(self.output_dim, 512),
                # self.net, nn.Linear(self.output_dim, 512),
                # self._node(threshold=self._threshold, decay=self._decay),
                self._node(threshold=self._threshold, v_reset=self.v_reset),
                # nn.Linear(512, np.prod(action_shape))
                nn.Linear(512, np.prod(action_shape), bias=False)
            )
            self.output_dim = np.prod(action_shape)

       
       
    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()
        
    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: x -> Q(x, \*)."""
        self.reset()
        # obs = torch.as_tensor(x, device=self.device, dtype=torch.float32) 
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32) / 255.0

        qs = []

        for i in range(self._time_window):
            value = self.net(x)
            qs.append(value)
        if self.features_only:
            return qs, state
        else:
            q_values = sum(qs) / self._time_window    
            return q_values, state
            