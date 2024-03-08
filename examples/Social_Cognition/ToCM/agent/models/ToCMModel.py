import torch
import torch.nn as nn

from environments import Env
from networks.ToCM.dense import DenseBinaryModel, DenseModel
from networks.ToCM.vae import Encoder, Decoder
from networks.ToCM.rnns import RSSMRepresentation, RSSMTransition

from thop import profile
from thop import clever_format

class ToCMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.action_size = config.ACTION_SIZE

        self.observation_encoder = Encoder(in_dim=config.IN_DIM, hidden=config.HIDDEN, embed=config.EMBED)  # in_dim:
        self.observation_decoder = Decoder(embed=config.FEAT, hidden=config.HIDDEN, out_dim=config.IN_DIM)

        self.transition = RSSMTransition(config, config.MODEL_HIDDEN)
        self.representation = RSSMRepresentation(config, self.transition)  # ann
        self.reward_model = DenseModel(config.FEAT, 1, config.REWARD_LAYERS, config.REWARD_HIDDEN)  # ann
        self.pcont = DenseBinaryModel(config.FEAT, 1, config.PCONT_LAYERS, config.PCONT_HIDDEN)

        if config.ENV_TYPE == Env.STARCRAFT:
            # print("config.FEAT, config.ACTION_SIZE, config.PCONT_LAYERS, config.PCONT_HIDDEN:", config.FEAT,
            #       config.ACTION_SIZE, config.PCONT_LAYERS, config.PCONT_HIDDEN)  # 1280 7 2 256
            self.av_action = DenseBinaryModel(config.FEAT, config.ACTION_SIZE, config.PCONT_LAYERS, config.PCONT_HIDDEN)
        else:
            self.av_action = None

        self.q_features = DenseModel(config.HIDDEN, config.PCONT_HIDDEN, 1, config.PCONT_HIDDEN)
        self.q_action = nn.Linear(config.PCONT_HIDDEN, config.ACTION_SIZE)

        # input_encoder = torch.randn(1, 10, config.IN_DIM)
        # macs, params = profile(self.observation_encoder, inputs=(input,))


    def forward(self, observations, prev_actions=None, prev_states=None, mask=None):
        if prev_actions is None:
            prev_actions = torch.zeros(observations.size(0), observations.size(1), self.action_size,
                                       device=observations.device)

        if prev_states is None:
            prev_states = self.representation.initial_state(prev_actions.size(0), observations.size(1),
                                                            device=observations.device)

        return self.get_state_representation(observations, prev_actions, prev_states, mask)

    def get_state_representation(self, observations, prev_actions, prev_states, mask):
        """
        :param observations: size(batch, n_agents, in_dim)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState
        """
        # print("mask = ", mask)
        obs_embeds = self.observation_encoder(observations)
        # print("obs_embeds=", obs_embeds)
        _, states = self.representation(obs_embeds, prev_actions, prev_states, mask)
        # print("state = ", states)
        return states
