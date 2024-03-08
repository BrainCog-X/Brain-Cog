import numpy as np
import torch
import wandb
import torch.nn.functional as F

from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, log_prob_loss, info_loss
from agent.utils.params import FreezeParameters
from networks.ToCM.rnns import rollout_representation, rollout_policy


def model_loss(config, model, obs, action, av_action, reward, done, fake, last):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1)
    feat_dec = post.get_features()
    # decoder inputs:reshape obs[:-1] to self.SEQ_LENGTH * self.MODEL_BATCH_SIZE, n_agents, dim
    reconstruction_loss, i_feat = rec_loss(model.observation_decoder,   # decoder
                   feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),  # input of decoder
                   obs[:-1].reshape(-1, n_agents, obs.shape[-1]),   # label real obs
                   1. - fake[:-1].reshape(-1, n_agents, 1))  # fake
    reward_loss = F.smooth_l1_loss(model.reward_model(feat), reward[1:])    # reward
    # print("pcont_loss")
    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
    div = state_divergence_loss(prior, post, config)    #kl

    model_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss
    if np.random.randint(20) == 4:
        wandb.log({'Model/reward_loss': reward_loss, 'Model/div': div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss})

    return model_loss


def actor_rollout(obs, action, last, model, actor, critic, config):  # model=ToCMLearnerModel
    n_agents = obs.shape[2]  # 2
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))

        items = rollout_policy(model, model.av_action, config.HORIZON, actor, post, action, config)  # horizon is 15 TODO  av_action: 49 40 2 7
        #
    imag_feat = items["imag_states"].get_features()
    obs_pred = items["obs_preds"]   # TODO
    # old_policy = items['old_policy']
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    # obs_pred_rew = items["obs_preds"][:-1]  # TODO
    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)
    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(),  # TODO pi
              imag_feat[:-1].detach(),
              items["imag_states"].map(lambda x: x[:-1]),
              obs_pred[:-1].detach(),
              returns.detach()]

    return [batch_multi_agent(v, n_agents) for v in output]


def critic_rollout(model, critic, states, rew_states, actions, raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        # print("states:", states.shape)
        # print("actions: ", actions.shape)
        value = critic(states, actions)
        # print("discount_arr")
        discount_arr = model.pcont(rew_states).mean
        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                   'Value/Value': value.mean()})
    returns = compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA)
    return returns


def calculate_reward(model, states, mask=None):
    imag_reward = model.reward_model(states)
    if mask is not None:
        imag_reward *= mask
    return imag_reward


def calculate_next_reward(model, actions, states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    return calculate_reward(model, imag_rew_feat)


def actor_loss(model, imag_state, obs_pred, actions, av_actions, old_policy, advantage, actor, ent_weight, config):
    next_state = model.transition(actions, imag_state)  # TODO
    next_feat = next_state.get_features().detach()  # TODO
    observations_next_other, _ = model.observation_decoder(next_feat)  # TODO
    _, new_policy = actor(torch.cat((obs_pred, observations_next_other[:, :, -(config.num_agents-1)*4:-(config.num_agents-1)*2]), -1))  # TODO
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)
    rho = (F.log_softmax(new_policy, dim=-1).gather(2, actions) -  # new policy is PPO pi
           F.log_softmax(old_policy, dim=-1).gather(2, actions)).exp()  # old policy is actor_rollout pi
    ppo_loss, ent_loss = calculate_ppo_loss(new_policy, rho, advantage)  # normalized
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Mean action': actions.float().mean()})
    return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()


def value_loss(critic, actions, imag_feat, targets):
    value_pred = critic(imag_feat, actions)

    mse_loss = (targets - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)
