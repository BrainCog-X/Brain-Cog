import torch
from torch.optim import Adam
import torch.nn.functional as F
from gym.spaces import Box, Discrete, MultiDiscrete
from multiagent.multi_discrete import MultiDiscrete
from utils.networks import MLPNetwork, SNNNetwork, LSTMClassifier
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent
from algorithms.ToM_class import ToM1
# from commom.distributions import make_pdtype
from thop import profile
from thop import clever_format

import  time
MSELoss = torch.nn.MSELoss()
KL_criterion = torch.nn.KLDivLoss(reduction='sum')
CE_criterion = torch.nn.CrossEntropyLoss(reduction="sum")


class ToM_decision11(object):

    def __init__(self, agent_init_params, alg_types, agent_types, num_lm,
                 output_style, device, config, gamma=0.95, tau=0.01, lr=0.01,
                 hidden_dim=64, discrete_action=False):
        self.config = config
        self.device = device
        self.num_lm = num_lm
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agent_types = agent_types
        self.num_good_agents = len(self._get_index1(self.agent_types, 'agent'))
        self.agents = [DDPGAgent_ToM(lr=lr, discrete_action=discrete_action,
                                     hidden_dim=hidden_dim,
                                     **params, output_style=output_style,
                                     num_agents=self.nagents,
                                     device=self.device)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        # tom0
        self.mle_base = [MLPNetwork(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2 + 5,
                                  self.agent_init_params[-1]['num_out_pol'],
                                  hidden_dim=5, norm_in=False),    # infer good agent
                         MLPNetwork(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2 + 5,
                                    self.agent_init_params[-1]['num_out_pol'],
                                    hidden_dim=5, norm_in=False),  # infer adversary
                         ]

        self.tom_base = {
            'agent': {'agent': self.mle_base[0],
                      'adversary': self.mle_base[1]
                      },
            'adversary': {'agent': self.mle_base[0],
                          'adversary': self.mle_base[1]
                          }
        }

        self.tom_PHI = [LSTMClassifier(self.num_good_agents * 2 + self.num_lm * 2 +
                                    (self.nagents - 1) * 2 + 5 * (self.nagents - 1),
                                  self.agent_init_params[-1]['num_out_pol'],
                                  hidden_size=64),    # infer good agent
                         LSTMClassifier(self.num_good_agents * 2 + self.num_lm * 2 +
                                    (self.nagents - 1) * 2 + 5 * (self.nagents - 1),
                                    self.agent_init_params[-1]['num_out_pol'],
                                    hidden_size=64),  # infer adversary
                         ]   #TODO
        self._agent_tom_init()  #TODO

        self.tom1 = ToM1(self.tom_base, alg_types, agent_types, num_lm, device)
        self.actions_tom0 = []
        self.next_actions_tom0 = []
        self.actions_tom1 = []
        self.next_actions_tom1 = []
        self.mle_opts = [Adam(i.parameters(), lr=1e-4) for i in self.mle_base]
        self.PHI_opts = [Adam(i.parameters(), lr=1e-4) for i in self.tom_PHI]
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.mle_dev = 'cpu'
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def _get_index1(self, lst=None, item=''):
        return [index for (index, value) in enumerate(lst) if value == item]

    def _agent_tom_init(self):
        # other_alg_types_ = self.alg_types.copy()
        other_agent_types_ = self.agent_types.copy()
        for agent_i in range(self.nagents):
            # other_alg_types = other_alg_types_.copy()
            other_agent_types = other_agent_types_.copy()
            # other_alg_types.pop(agent_i)
            other_agent_types.pop(agent_i)

            adv_indx = self._get_index1(other_agent_types, 'adversary')
            good_indx = self._get_index1(other_agent_types, 'agent')
            self.agents[agent_i].mle += [self.tom_base[self.agent_types[agent_i]]['adversary']] * len(adv_indx)     #TODO
            self.agents[agent_i].mle += [self.tom_base[self.agent_types[agent_i]]['agent']] * len(good_indx)

    def step(self, observations, actions_pre, explore=False):    #simple_tag
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        # t1 = time.time()
        observations_ = observations.copy()
        actions_pre_ = actions_pre.copy()
        # other_alg_types_ = self.alg_types.copy()
        other_agent_types_ = self.agent_types.copy()
        adv_agent_indx = self._get_index1(self.agent_types, 'adversary')
        good_agent_indx = self._get_index1(self.agent_types, 'agent')
        '''
        tom0
        '''
        actions_tom0 = []
        actions_tom0 += [
            gumbel_softmax(
                self.mle_base[1].to(self.device)(
                    torch.cat((observations[j][:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                               actions_pre[j][:, :5]), 1).to(self.device)).detach(), hard=True
            ) for j in adv_agent_indx
        ]
        actions_tom0 += [
            gumbel_softmax(
                self.mle_base[0].to(self.device)(
                    torch.cat((observations[j][:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                               actions_pre[j][:, :5]), 1).to(self.device)).detach(), hard=True
            ) for j in good_agent_indx
        ]

        '''
        tom1
        '''
        actions_tom1 = []
        for agent_i, obs in enumerate(observations):
            obs_ = observations_.copy()
            acs_other = actions_tom0.copy()
            other_agent_types = other_agent_types_.copy()
            obs_.pop(agent_i)
            acs_other.pop(agent_i)
            other_agent_types.pop(agent_i)

            if agent_i in adv_agent_indx:
                actions_tom1.append(
                    gumbel_softmax(
                        self.tom_PHI[1].to(self.device)(
                    torch.cat((obs[:, -(self.num_good_agents * 2 + self.num_lm * 2 +
                    (self.nagents - 1) * 2):].to(self.device), torch.cat(acs_other, 1)), 1)), hard=True
                    ).cpu()
                )

            elif agent_i in good_agent_indx:
                actions_tom1.append(
                    gumbel_softmax(
                        self.tom_PHI[0].to(self.device)(
                    torch.cat((obs[:, -(self.num_good_agents * 2 + self.num_lm * 2 +
                    (self.nagents - 1) * 2):].to(self.device), torch.cat(acs_other, 1)), 1)), hard=True
                    ).cpu()
                )


        observations = self._get_obs(observations, actions_tom1)

        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def _get_obs(self, observations, action_tom):
        observations_ = []
        other_actions_tom_ = action_tom.copy()
        for agent_i, obs in enumerate(observations):
            other_action_tom = other_actions_tom_.copy()
            other_action_tom.pop(agent_i)
            actions = other_action_tom
            observations_.append(torch.cat((obs, torch.cat(actions, 1)), 1))
        return observations_

    def train_tom0(self, sample, agent_i):
        acs_pre, obs, acs, rews, next_obs, dones = sample

        adv_agent_indx = self._get_index1(self.agent_types, 'adversary')
        good_agent_indx = self._get_index1(self.agent_types, 'agent')
        # self.agent_types[tom_agent_indx]
        '''
        data
        for with_tom
        for without_tom
        '''
        if adv_agent_indx != []:
            adv_input = torch.cat([torch.cat((obs[i], acs_pre[i][:, :5]), 1) for i in adv_agent_indx])
            label_adv_output = torch.cat([acs[i][:, :5] for i in adv_agent_indx])
            self.mle_base[1].zero_grad()
            adv_output = self.mle_base[1](adv_input[:,
                      -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2 + 5):])
            loss_adv = F.mse_loss(adv_output.float(), label_adv_output.float())
            loss_adv.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

        if good_agent_indx != []:
            good_input = torch.cat([torch.cat((obs[i], acs_pre[i][:, :5]), 1) for i in good_agent_indx])
            label_good_output = torch.cat([acs[i][:, :5] for i in good_agent_indx])
            self.mle_base[0].zero_grad()
            good_output = self.mle_base[0](good_input[:,
                      -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2 + 5):])
            loss_good = F.mse_loss(good_output.float(), label_good_output.float())
            loss_good.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()
        '''
        Only train agents with ToM (adversarys)
        '''
        '''
        adv-adv
        adv-good
        '''

    def tom1_infer_other(self, sample):
        acs_pre, obs, acs, rews, next_obs, dones = sample
        self.actions_tom1 = []
        self.next_actions_tom1 = []
        actions_tom1 = []
        next_actions_tom1 = []
        other_actions_tom0_ = self.actions_tom0.copy()
        other_next_actions_tom0_ = self.next_actions_tom0.copy()
        adv_agent_indx = self._get_index1(self.agent_types, 'adversary')
        good_agent_indx = self._get_index1(self.agent_types, 'agent')
        good_in = []
        adv_in =[]
        good_in_next = []
        adv_in_next =[]
        for agent_i, (obs_i, next_obs_i) in enumerate(zip(obs, next_obs)):
            other_action_tom0 = other_actions_tom0_.copy()
            other_next_actions_tom0 = other_next_actions_tom0_.copy()
            other_action_tom0.pop(agent_i)
            other_next_actions_tom0.pop(agent_i)
            if agent_i in adv_agent_indx:
                adv_in.append(torch.cat(
                    (obs_i[:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                     torch.cat(other_action_tom0, 1)), 1))
                adv_in_next.append(torch.cat(
                    (next_obs_i[:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                     torch.cat(other_next_actions_tom0, 1)), 1))

            elif agent_i in good_agent_indx:
                good_in.append(torch.cat(
                    (obs_i[:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                     torch.cat(other_action_tom0, 1)), 1))
                good_in_next.append(torch.cat(
                    (next_obs_i[:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                     torch.cat(other_next_actions_tom0, 1)), 1))

        if adv_agent_indx != []:
            adv_in = torch.cat(adv_in, 0)
            adv_in_next = torch.cat(adv_in_next, 0)
            actions_tom1.append(gumbel_softmax(
                self.tom_PHI[1].to(self.device)(
                    adv_in), hard=True
            ))

            next_actions_tom1.append(gumbel_softmax(
                self.tom_PHI[1].to(self.device)(
                    adv_in_next), hard=True
            ))  # adv
            label_adv_output = torch.cat([self.actions_tom0[i] for i in adv_agent_indx]).detach()
            # label_adv_output = torch.cat([acs[i][:, :5] for i in adv_agent_indx])
            adv_output = actions_tom1[0]
            loss_adv = F.mse_loss(adv_output.float(), label_adv_output.float())
            loss_adv.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.PHI_opts[1].step()
        if good_agent_indx != []:
            good_in = torch.cat(good_in, 0)
            good_in_next = torch.cat(good_in_next, 0)
            actions_tom1.append(gumbel_softmax(
                self.tom_PHI[0].to(self.device)(
                    good_in), hard=True
            ))
            next_actions_tom1.append(gumbel_softmax(
                self.tom_PHI[0].to(self.device)(
                    good_in_next), hard=True
            ))  # agent
            label_good_output = torch.cat([self.actions_tom0[i] for i in good_agent_indx]).detach()
            # label_good_output = torch.cat([acs[i][:, :5] for i in good_agent_indx])
            if self.config.env_id == 'simple_spread' or self.config.env_id == 'hetero_spread':
                good_output = actions_tom1[0]
            else:
                good_output = actions_tom1[1]
            loss_good = F.mse_loss(good_output.float(), label_good_output.float())
            loss_good.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.PHI_opts[0].step()

        actions_tom1 = torch.cat(actions_tom1)#.detach()
        next_actions_tom1 = torch.cat(next_actions_tom1).detach()
        for i in range(self.nagents):
            self.actions_tom1.append(actions_tom1[i*self.config.batch_size:(i+1)*self.config.batch_size, :])
            self.next_actions_tom1.append(next_actions_tom1[i*self.config.batch_size:(i+1)*self.config.batch_size, :])
        # print(self.actions_tom1)

    def tom0_output(self, sample):
        acs_pre, obs, acs, rews, next_obs, dones = sample
        self.actions_tom0 = []
        self.next_actions_tom0 = []
        adv_indx = self._get_index1(self.agent_types, 'adversary')
        good_indx = self._get_index1(self.agent_types, 'agent')
        self.actions_tom0 += [
            gumbel_softmax(
                self.mle_base[1].to(self.device)(
torch.cat((obs[j][:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                               acs_pre[j][:, :5]), 1)).detach(), hard=True
            ) for j in adv_indx
        ]
        self.actions_tom0 += [
            gumbel_softmax(
                self.mle_base[0].to(self.device)(
torch.cat((obs[j][:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                               acs_pre[j][:, :5]), 1)).detach(), hard=True
            ) for j in good_indx
        ]
        self.next_actions_tom0 += [
            gumbel_softmax(
                self.mle_base[1].to(self.device)(
torch.cat((next_obs[j][:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                               acs[j][:, :5]), 1)).detach(), hard=True
            ) for j in adv_indx
        ]
        self.next_actions_tom0 += [
            gumbel_softmax(
                self.mle_base[0].to(self.device)(
torch.cat((next_obs[j][:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                               acs[j][:, :5]), 1)).detach(), hard=True
            ) for j in good_indx
        ]
        # actions_tom0 = torch.cat(actions_tom0, 1)
        # return actions_tom0, next_actions_tom0

    def update(self, sample, agent_i, parallel=False, logger=None, sample_r=None):

        # other_alg_types = self.alg_types.copy()
        other_agent_types = self.agent_types.copy()
        # other_alg_types.pop(agent_i)
        other_agent_types.pop(agent_i)
        adv_indx = self._get_index1(other_agent_types, 'adversary')
        good_indx = self._get_index1(other_agent_types, 'agent')
        agent_i_alg = self.alg_types[agent_i]
        acs_pre, obs, acs, rews, next_obs, dones = sample

        next_obs_ = self._get_obs(next_obs, self.next_actions_tom1)
        obs_ = self._get_obs(obs, self.actions_tom1)
        curr_agent = self.agents[agent_i]

        '''
        distance between self and other
        '''
        Euclidean_D = []
        for i in range(len(other_agent_types)):
            Euclidean_D.append(obs[agent_i][:,
               -len(other_agent_types)*2:][:, i: i+2].pow(2).sum(1).sqrt())
        Euclidean_D_ = torch.stack(Euclidean_D, 1)
        '''
        distance between self and landmark
        '''
        Euclidean_L = []
        for i in range(self.num_lm):
            Euclidean_L.append(obs[agent_i][:, -len(other_agent_types)*2-self.num_lm*2:
                   -len(other_agent_types)*2][:, i: i+2].pow(2).sum(1).sqrt())
        Euclidean_L = torch.stack(Euclidean_L, 1)

        close_agent_index = (Euclidean_D_ == Euclidean_D_.min(dim=1, keepdim=True)[0])\
            .to(dtype=torch.int32)    #run11/run12 self-orgnization
        # close_agent_index = torch.ones((self.config.batch_size, len(other_agent_types))) \
        #         .to(dtype=torch.int32).to(self.device)  #run13

        if agent_i == 0:
            self.train_tom0(sample, agent_i)

        E_action = self.tom1.tom1_output(agent_i, adv_indx,
          good_indx, obs[agent_i], acs_pre[agent_i])

        if agent_i_alg == 'with_tom':
            acs_other = acs.copy()
            acs_other.pop(agent_i)

            # KL loss
            # adv_loss = sum([KL_criterion(E_action[j], acs_other[j][:, :5].float()) for j in adv_indx])  #TODO
            # good_loss = sum([KL_criterion(E_action[j], acs_other[j][:, :5].float()) for j in good_indx])

            # L2 loss
            action_loss = torch.norm(acs[agent_i][:, :5] - E_action[0], p=2, dim=1)
            loss_other = 0.1 * torch.stack([action_loss]*len(other_agent_types), 1)

            if agent_i in adv_indx:
                '''
                adv_loss : decrease
                good_loss : increase
                '''
                close_agent_index[:, good_indx] *= -1
                close_agent_index[:, adv_indx] *= 0.1
                intri_rew = close_agent_index.mul(loss_other).mul(Euclidean_D_).sum(1)
                # intri_rew = close_agent_index.mul(loss_other).sum(1)
            else:
                '''
                adv_loss : increase
                good_loss : decrease
                '''
                close_agent_index[:, adv_indx] *= 1
                close_agent_index[:, good_indx] *= 0.1
                # if self.config.env_id == 'simple_adversary':
                #     intri_rew = close_agent_index.mul(loss_other).mul(Euclidean_D_).sum(1) - \
                #                 obs[agent_i][:, :2].pow(2).sum(1)
                # elif self.config.env_id == 'simple_spread_pre':
                #     intri_rew = close_agent_index.mul(loss_other).mul(Euclidean_D_).sum(1) - \
                #                 Euclidean_L.min(dim=1, keepdim=True)[0][:,0]
                # else:
                intri_rew = close_agent_index.mul(loss_other).mul(Euclidean_D_).sum(1)
                    # intri_rew = close_agent_index.mul(loss_other).sum(1)
            rews[agent_i] = rews[agent_i] + intri_rew.detach()


        # center critic
        curr_agent.critic_optimizer.zero_grad()
        all_trgt_acs = []
        if self.discrete_action:  # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs_)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        vf_in = torch.cat((*obs, *acs), dim=1)

        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.

            curr_pol_out = curr_agent.policy(obs_[agent_i].detach())
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs_):
            ob = ob.detach()
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out ** 2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        # actor
        curr_agent.policy_optimizer.step()
        # print('c_loss:',vf_loss, 'p_loss:', pol_loss)
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        # for i in self.tom_base.values():
        #     for mle in i.values():
        #         mle.train()
        for mle in self.mle_base:
            mle.train()
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
            for mle_i in a.mle:
                mle_i.train()
        if device == 'gpu':
            fn = lambda x: x.to(torch.device(self.device))
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device
        if not self.mle_dev == device:
            # for i in self.tom_base.keys():
            #     for j in self.tom_base[i].keys():
            #         self.tom_base[i][j] = fn(mle)
            for i, mle in enumerate(self.mle_base):
                self.mle_base[i] = fn(mle)
            for a in self.agents:
                for i, mle_i in enumerate(a.mle):
                    a.mle[i] = fn(mle_i)
            self.mle_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.to(torch.device(self.device))
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'tom_params': [self.get_params()],}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, config, device, agent_alg, adversary_alg,
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, output_style='sum'):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        num_lm = env.num_lm
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
            num_in_mle = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            elif isinstance(acsp, Discrete):  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            elif isinstance(acsp, MultiDiscrete):
                discrete_action = True
                get_shape = lambda x: sum(x.high - x.low + 1)
            num_out_pol = get_shape(acsp)
            # if algtype == "with_tom":
            num_in_critic = 0
            num_in_pol += (len(env.agent_types)-1) * 5
            for oobsp in env.observation_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in env.action_space:
                if isinstance(oacsp, Box):
                    discrete_action = False
                    get_shape = lambda x: x.shape[0]
                elif isinstance(oacsp, Discrete):  # Discrete
                    discrete_action = True
                    get_shape = lambda x: x.n
                elif isinstance(oacsp, MultiDiscrete):
                    discrete_action = True
                    get_shape = lambda x: sum(x.high - x.low + 1)
                num_in_critic += get_shape(oacsp)
            # else:
            #     num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'num_in_mle': num_in_mle,})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'device': device,
                     'config' : config,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_types' : env.agent_types,
                     'num_lm' : num_lm,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'output_style': output_style}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        for a, params in zip([instance], save_dict['tom_params']):
            a.load_params(params)
        return instance


    def get_params(self):
        params = {
                }
        for i in range(len(self.mle_base)):
            params['mle%d'%i] = self.mle_base[i].state_dict()
            params['mle_optimizer%d'%i] = self.mle_opts[i].state_dict()
            params['tom_phi%d'%i] = self.tom_PHI[i].state_dict()
            params['phi_opt%d' % i] = self.PHI_opts[i].state_dict()
        return params

    def load_params(self, params):
        for i in range(len(self.mle_base)):
            self.mle_base[i].load_state_dict(params['mle%d'%i])
            self.mle_opts[i].load_state_dict(params['mle_optimizer%d'%i])
            self.tom_PHI[i].load_state_dict(params['tom_phi%d'%i])
            self.PHI_opts[i].load_state_dict(params['phi_opt%d' % i])
