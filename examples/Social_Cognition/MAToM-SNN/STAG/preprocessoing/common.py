"""
preprocess
"""

from typing import Union
import abc

import torch as tc
import numpy as np

class Preprocessing(abc.ABC, tc.nn.Module):
    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Creates an input vector for a meta-learning agent.

        Args:
            curr_obs: either tc.LongTensor or tc.FloatTensor of shape [B, ...].
            prev_action: tc.LongTensor of shape [B, ...]
            prev_reward: tc.FloatTensor of shape [B, ...]
            prev_done: tc.FloatTensor of shape [B, ...]

        Returns:
            tc.FloatTensor of shape [B, ..., ?]
        """
        pass


def one_hot_torch(ys: tc.LongTensor, depth: int, device) -> tc.FloatTensor:
    """
    Applies one-hot encoding to a batch of vectors.

    Args:
        ys: tc.LongTensor of shape [B].
        depth: int specifying the number of possible y values.

    Returns:
        the one-hot encodings of tensor ys.
    """

    vecs_shape = list(ys.shape) + [depth]
    vecs = tc.zeros(dtype=tc.float32, size=vecs_shape).to(device)
    vecs.scatter_(dim=-1, index=ys.unsqueeze(-1),
                  src=tc.ones(dtype=tc.float32, size=vecs_shape).to(device))
    return vecs.float()


def one_hot(ys: int, depth: int) -> list:
    """
    Applies one-hot encoding to a batch of vectors.

    Args:
        ys: tc.LongTensor of shape [B].
        depth: int specifying the number of possible y values.

    Returns:
        the one-hot encodings of tensor ys.
    """

    letter = [0 for _ in range(depth)]
    letter[ys-1] = 1
    letter = np.array(letter)
    # print(letter)
    return letter
