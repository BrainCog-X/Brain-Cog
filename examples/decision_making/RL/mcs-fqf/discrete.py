from audioop import bias
from time import time
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch
from braincog.base.node.node import LIFNode, ThreeCompNode

class SpikePopEncodingNetwork(nn.Module):
    """Cosine embedding network for IQN. Convert a scalar in [0, 1] to a list \
    of n-dim vectors.

    :param num_cosines: the number of cosines used for the embedding.
    :param embedding_dim: the dimension of the embedding/output.

    .. note::

        From https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_cosines: int, embedding_dim: int, device, time_window: int=8) -> None:
        super().__init__()
        self._threshold = 0.5
        # self._decay = 0.2
        self._decay = 0.5
        self.r_max = 0.5
        # self.mus = torch.from_numpy(np.arange(num_cosines) / num_cosines)
        
        self.sigma = 0.05
        # self.sigma = 0.01
        self._node = LIFNode
        self.net = nn.Sequential(
            nn.Linear(num_cosines, embedding_dim), 
            self._node()
            # self._node(threshold=self._threshold, decay=self._decay)
            )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim
        self.mus = torch.arange(0, num_cosines, device=device).view(1, 1, self.num_cosines) / num_cosines

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

    def forward(self, taus: torch.Tensor, time_window: int) -> torch.Tensor:
        batch_size = taus.shape[0]
        N = taus.shape[1]
        self.reset()


        taus_lam = self.r_max * torch.exp(-(taus.unsqueeze(-1) - self.mus)**2/2/self.sigma**2).view(batch_size*N, self.num_cosines)
        taus_repeat = taus_lam.unsqueeze(0).repeat(time_window, 1, 1)
        taus_emb = torch.poisson(taus_repeat)
      
        tau_embeddings = []
        
        for i in range(time_window):
            t_e = self.net(taus_emb[i])
            tau_embeddings.append(t_e)
        return tau_embeddings   

class SpikeFractionProposalNetwork(nn.Module):
    """Fraction proposal network for FQF.

    :param num_fractions: the number of factions to propose.
    :param embedding_dim: the dimension of the embedding/input.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_fractions: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(embedding_dim, num_fractions)
        torch.nn.init.xavier_uniform_(self.net.weight, gain=0.01)
        torch.nn.init.constant_(self.net.bias, 0)
        self.num_fractions = num_fractions
        self.embedding_dim = embedding_dim

    def forward(
        self, state_embeddings: list
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_embeddings = torch.stack(state_embeddings).detach() 
        
        time_window = state_embeddings.shape[0]
        batch_size = state_embeddings.shape[1]
        logits = self.net(state_embeddings.view(time_window*batch_size, -1))
        logits = logits.view(time_window, batch_size, -1)
        m = torch.distributions.Categorical(logits=logits.mean(0))
        taus_1_N = torch.cumsum(m.probs, dim=1)
        taus = F.pad(taus_1_N, (1, 0))
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0
        entropies = m.entropy()
        return taus, tau_hats, entropies 


class MCQuantiles(nn.Module):
    def __init__(self, 
            state_embedings_shape: int, 
            tau_embeddings_shape: int,
            hidden_size: int, 
            last_size: int,
            fusion_size : int=512,
            tau_s : int = 2.0):
        super().__init__()
       
        self.basal_w = nn.Linear(state_embedings_shape, fusion_size, bias=False)
        self.apical_w = nn.Linear(tau_embeddings_shape, fusion_size, bias=False)
        self._node = LIFNode
        self.mc_node = ThreeCompNode()
        
        self._last = nn.Sequential(
            nn.Linear(fusion_size, hidden_size),           
            self._node(),
            nn.Linear(hidden_size, last_size),
        )

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()
    
    def forward(self, state_embedding, tau_embedding):
        """
        state_embedding: list
        tau_embedding: torch.Tensor
        """
        self.reset()
        
        assert isinstance(state_embedding, type(tau_embedding))
        if isinstance(state_embedding, list):
            time_window = len(state_embedding) 
            
        elif isinstance(state_embedding, torch.Tensor):
            time_window = state_embedding.shape[0]
        else:
            raise TypeError('Not support data type.')
        batch_size = state_embedding[0].shape[0]
        sample_size = tau_embedding[0].shape[0] // batch_size 


        quantiles = []
      
        for step in range(time_window):
           
            basal_psp = self.basal_w(state_embedding[step]).unsqueeze(1)     
            apical_psp = self.apical_w(tau_embedding[step]).view(batch_size, sample_size, -1)  
            embeddings = self.mc_node({'basal_inputs': basal_psp, 'apical_inputs': apical_psp}).view(batch_size*sample_size, -1) 
            out = self._last(embeddings)    
            quantiles.append(out)
        
        quantiles = sum(quantiles) / time_window
        return quantiles.view(batch_size, sample_size, -1).transpose(1, 2)

   
    
class SpikeFullQuantileFunction(nn.Module):
    """Full(y parameterized) Quantile Function.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_dim: the dimension of action space.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        The first return value is a tuple of (quantiles, fractions, quantiles_tau),
        where fractions is a Batch(taus, tau_hats, entropies).
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.last_size = np.prod(action_shape)
        self.preprocess = preprocess_net
        self.input_dim = getattr(
            self.preprocess, "output_dim", preprocess_net_output_dim
        )
        self.embed_model = SpikePopEncodingNetwork(num_cosines,
                                                  self.input_dim, device=device).to(device)
        self.mcquantiles = MCQuantiles(self.input_dim, self.input_dim, hidden_size=np.prod(hidden_sizes),
                                     last_size=action_shape).to(device)
    def forward(  # type: ignore
        self, s: Union[np.ndarray, torch.Tensor],
        propose_model: SpikeFractionProposalNetwork,
        fractions: Optional[Batch] = None,
        **kwargs: Any
    ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state=kwargs.get("state", None))  
        # Propose fractions
        if fractions is None:
            taus, tau_hats, entropies = propose_model(logits)
            fractions = Batch(taus=taus, tau_hats=tau_hats, entropies=entropies)
        else:
            taus, tau_hats = fractions.taus, fractions.tau_hats

        time_window = len(logits)
        tau_hats_emb = self.embed_model(tau_hats, time_window)
        
        quantiles = self.mcquantiles(logits, tau_hats_emb)
        
        quantiles_tau = None
        if self.training:
            with torch.no_grad():
                tau_emb = self.embed_model(taus[:, 1:-1], time_window)        
                quantiles_tau = self.mcquantiles(logits, tau_emb)
        return (quantiles, fractions, quantiles_tau), h



