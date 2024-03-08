import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical

from configs.ToCM.ToCMAgentConfig import RSSMState
from networks.transformer.layers import AttentionEncoder


def stack_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.stack)


def cat_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.cat)


def reduce_states(rssm_states: list, dim, func):
    return RSSMState(*[func([getattr(state, key) for state in rssm_states], dim=dim)
                       for key in rssm_states[0].__dict__.keys()])


class DiscreteLatentDist(nn.Module):
    def __init__(self, in_dim, n_categoricals, n_classes, hidden_size):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.dists = nn.Sequential(nn.Linear(in_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, n_classes * n_categoricals))

    def forward(self, x):
        logits = self.dists(x).view(x.shape[:-1] + (self.n_categoricals, self.n_classes))
        class_dist = OneHotCategorical(logits=logits)
        one_hot = class_dist.sample()
        latents = one_hot + class_dist.probs - class_dist.probs.detach()
        return logits.view(x.shape[:-1] + (-1,)), latents.view(x.shape[:-1] + (-1,))


class RSSMTransition(nn.Module):
    def __init__(self, config, hidden_size=200, activation=nn.ReLU):
        super().__init__()
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self._hidden_size = hidden_size
        self._activation = activation
        self._cell = nn.GRU(hidden_size, self._deter_size)
        self._attention_stack = AttentionEncoder(3, hidden_size, hidden_size, dropout=0.1)
        self._rnn_input_model = self._build_rnn_input_model(config.ACTION_SIZE + self._stoch_size)
        self._stochastic_prior_model = DiscreteLatentDist(self._deter_size, config.N_CATEGORICALS, config.N_CLASSES,
                                                          self._hidden_size)

    def _build_rnn_input_model(self, in_dim):
        rnn_input_model = [nn.Linear(in_dim, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def forward(self, prev_actions, prev_states, mask=None):
        batch_size = prev_actions.shape[0]
        n_agents = prev_actions.shape[1]
        stoch_input = self._rnn_input_model(torch.cat([prev_actions, prev_states.stoch], dim=-1))
        attn = self._attention_stack(stoch_input, mask=mask)
        deter_state = self._cell(attn.reshape(1, batch_size * n_agents, -1),
                                 prev_states.deter.reshape(1, batch_size * n_agents, -1))[0].reshape(batch_size, n_agents, -1)
        logits, stoch_state = self._stochastic_prior_model(deter_state)
        return RSSMState(logits=logits, stoch=stoch_state, deter=deter_state)


class RSSMRepresentation(nn.Module):
    def __init__(self, config, transition_model: RSSMTransition):
        super().__init__()
        self._transition_model = transition_model
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self._stochastic_posterior_model = DiscreteLatentDist(self._deter_size + config.EMBED, config.N_CATEGORICALS,
                                                              config.N_CLASSES, config.HIDDEN)

    def initial_state(self, batch_size, n_agents, **kwargs):
        return RSSMState(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs))

    def forward(self, obs_embed, prev_actions, prev_states, mask=None):
        """
        :param obs_embed: size(batch, n_agents, obs_size)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState, global_state: size(batch, 1, global_state_size)
        """
        prior_states = self._transition_model(prev_actions, prev_states, mask)
        x = torch.cat([prior_states.deter, obs_embed], dim=-1)
        logits, stoch_state = self._stochastic_posterior_model(x)
        posterior_states = RSSMState(logits=logits, stoch=stoch_state, deter=prior_states.deter)
        return prior_states, posterior_states


def rollout_representation(representation_model, steps, obs_embed, action, prev_states, done):
    """
        Roll out the model with actions and observations from data.
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, n_agents, embedding_size)
        :param action: size(time_steps, batch_size, n_agents, action_size)
        :param prev_states: RSSM state, size(batch_size, n_agents, state_size)
        :return: prior, posterior states. size(time_steps, batch_size, n_agents, state_size)
        """
    priors = []
    posteriors = []
    for t in range(steps):
        prior_states, posterior_states = representation_model(obs_embed[t], action[t], prev_states)
        prev_states = posterior_states.map(lambda x: x * (1.0 - done[t]))
        priors.append(prior_states)
        posteriors.append(posterior_states)

    prior = stack_states(priors, dim=0)
    post = stack_states(posteriors, dim=0)
    return prior.map(lambda x: x[:-1]), post.map(lambda x: x[:-1]), post.deter[1:]


def rollout_policy(transition_model, av_action, steps, policy, prev_state, prev_action, config):  # av_action.shape=[49 40 2 7] policy=actor
    """
        Roll out the model with a policy function.
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
    state = prev_state
    action = prev_action[:-1].reshape((prev_action.shape[0] - 1) * prev_action.shape[1], prev_action.shape[2], -1)   # TODO
    next_states = []
    actions = []
    av_actions = []
    policies = []
    obs_preds = []
    for t in range(steps):
        feat = state.get_features().detach()
        obs_pred, _ = transition_model.observation_decoder(feat)  # TODO
        next_state = transition_model.transition(action, state)  # TODO
        next_feat = next_state.get_features().detach()  # TODO
        observations_next_other, _ = transition_model.observation_decoder(next_feat)  # TODO
        action, pi = policy(torch.cat((obs_pred.detach(), observations_next_other[:, :, -(config.num_agents-1)*4:-(config.num_agents-1)*2]), -1))  # TODO 3 vs 3

        # print("feat:", feat)
        # print("feat_shape:", feat.shape)  # feat.shape=1920,2,1280
        # action, pi = policy(feat)
        if av_action is not None:
            # print("av_action!")
            avail_actions = av_action(feat).sample()
            pi[avail_actions == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)
            av_actions.append(avail_actions.squeeze(0))
        next_states.append(state)
        obs_preds.append(obs_pred)
        policies.append(pi)
        actions.append(action)
        state = transition_model.transition(action, state)
    return {"imag_states": stack_states(next_states, dim=0),
            "obs_preds": torch.stack(obs_preds, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "old_policy": torch.stack(policies, dim=0)}
