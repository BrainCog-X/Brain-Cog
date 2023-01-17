import torch
from torch.optim import Adam
import torch.nn.functional as F
from gym.spaces import Box, Discrete, MultiDiscrete
from multiagent.multi_discrete import MultiDiscrete
from utils.networks import MLPNetwork, SNNNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from agents.agents import DDPGAgent, DDPGAgent_RNN, DDPGAgent_SNN, DDPGAgent_ToM
# from commom.distributions import make_pdtype


import  time
MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types, device,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.device = device
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
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

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
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
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
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
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
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
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, device, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
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
            if algtype == "MADDPG":
                num_in_critic = 0
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
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'device': device}
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
        return instance

class MADDPG_SNN(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,output_style, device,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.device = device
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent_SNN(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params, output_style=output_style)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
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

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG_SNN':
            all_trgt_acs = []
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
                # for nobs in next_obs:
                #     if nobs.shape[1] == next_obs[agent_i].shape[1]:
                #         all_trgt_acs.append(onehot_from_logits(self.target_policies[agent_i](nobs)))
                #     else:
                #         if next_obs[agent_i].shape[1] - nobs[:][3].shape[0] > 0 :
                #             a = torch.zeros((nobs.shape[0], next_obs[agent_i].shape[1] - nobs[:][3].shape[0]))
                #             a = a.to(torch.device(self.device))
                #             obs_good = torch.cat((nobs, a), 1)
                #             all_trgt_acs.append(onehot_from_logits(self.target_policies[agent_i](obs_good)))
                #         else:
                #             all_trgt_acs.append(onehot_from_logits(self.target_policies[agent_i](nobs[:, :next_obs[agent_i].shape[1]])))
                # all_trgt_acs = [onehot_from_logits(self.target_policies[agent_i](nobs)) for nobs in
                #                 next_obs]
            else:
                # all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                #                                              next_obs)]
                all_trgt_acs = [self.target_policies[agent_i](nobs) for nobs in next_obs]   #self-experience
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG_SNN':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
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
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG_SNN':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
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
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
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
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, device, agent_alg="MADDPG_SNN", adversary_alg="MADDPG_SNN",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, output_style='sum'):
    # def init_from_env(cls, env, agent_alg="MADDPG_SNN", adversary_alg="MADDPG_SNN",
    #                   gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):    #eval
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            num_in_pol = obsp.shape[0]
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
            if algtype == "MADDPG_SNN":
                num_in_critic = 0
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
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'output_style': output_style,
                     'device': device}
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
        return instance

class MADDPG_ToM(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, agent_init_params, alg_types, output_style, device,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.device = device
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent_ToM(lr=lr, discrete_action=discrete_action,
                                     hidden_dim=hidden_dim,
                                     **params, output_style=output_style,
                                     num_agents=self.nagents,
                                     device=self.device)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        if self.nagents == 6:
            self.mle_base = [SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14,      #simple_com
                                        self.agent_init_params[3]['num_out_pol'], #adv self-self
                                  hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14,
                                        self.agent_init_params[3]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             # SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14,
                             #            self.agent_init_params[3]['num_out_pol'],  # adv self-other
                             #            hidden_dim=hidden_dim, output_style=output_style),
                             # SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14,
                             #            self.agent_init_params[3]['num_out_pol'],
                             #            hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                             ]
        if self.nagents == 4:
            self.mle_base = [SNNNetwork(self.agent_init_params[0]['num_in_mle'] - 2,      #simple_tag
                                        self.agent_init_params[0]['num_out_pol'], #adv self-self
                                  hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 2,
                                        self.agent_init_params[3]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 2,
                                        self.agent_init_params[3]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                             ]
        elif self.nagents == 3:
            self.mle_base = [SNNNetwork(self.agent_init_params[1]['num_in_mle'],      #simple_adv
                                        self.agent_init_params[1]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle'],
                                        self.agent_init_params[1]['num_out_pol'], #agent self-self
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle'],
                                        self.agent_init_params[1]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
            ]
        elif self.nagents == 2:
            self.mle_base = [SNNNetwork(self.agent_init_params[0]['num_in_mle']-2,      #simple_push
                                        self.agent_init_params[0]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle']-2,
                                        self.agent_init_params[1]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                ]
        self.mle_opts = [Adam(self.mle_base[i].parameters(), lr=lr) for i in range(len(self.mle_base))]
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

    def step(self, observations, explore=False):    #simple_tag
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
        for agent_i, obs in enumerate(observations):
            obs_ = observations_.copy()
            obs_.pop(agent_i)
            # actions = [self.agents[agent_i].mle[j].cpu()(observations[agent_i]) for j, obs_j in enumerate(obs_)]
            # observations[agent_i] = torch.cat((observations[agent_i], torch.cat(actions, 1)), 1)
            if self.nagents == 6:

                if agent_i < 4:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0],self.mle_base[0], self.mle_base[1], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(obs_j[:, 4:24].to(self.device)),
                                              hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]

                else:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:-2]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(obs_j[:, 4:24].to(self.device)),
                                              hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]
            if self.nagents == 4:
                if agent_i < 3:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(obs_j[:, 2:].to(self.device)),
                                              hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]

                elif agent_i == 3:
                    self.agents[agent_i].mle = [self.mle_base[2],self.mle_base[2], self.mle_base[2]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:-2]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(obs_j[:,2:-2].to(self.device)),
                                              hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]
            elif self.nagents == 3: #simple_adv
                if agent_i < 1:
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0]]
                    # actions = [gumbel_softmax(
                    #     self.agents[agent_i].mle[j].cpu()(torch.cat((obs_j[:, :2], observations_[agent_i]), 1)),
                    #     hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:, :2],
                             observations_[agent_i]), 1).to(self.device)), hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]
                elif agent_i >= 1:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(observations_[agent_i].to(self.device)),
                                              hard=True).cpu() for j, obs_j in enumerate(obs_)]
            elif self.nagents == 2:
                if agent_i < 1:
                    self.agents[agent_i].mle = [self.mle_base[0]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i][:,2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(observations_[agent_i][:,2:].to(self.device)),
                                              hard=True).cpu() for j, obs_j in enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i][:, 2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(observations_[agent_i][:,2:].to(self.device)),
                                              hard=True).cpu() for j, obs_j in enumerate(obs_)]
            observations[agent_i] = torch.cat((observations[agent_i], torch.cat(actions, 1)), 1)
        # t2 = time.time()
        # print('step+time:', t2 - t1)
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]


    def _get_obs(self, observations):
        observations_ = []
        for agent_i, obs in enumerate(observations):
            obs_ = observations.copy()
            obs_.pop(agent_i)
            if self.nagents == 6:
                if agent_i < 4:   #simple_tag
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[1], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(obs_j[:, 4:24]).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
                elif agent_i > 4:
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(obs_j[:, 4:24]).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
            if self.nagents == 4:
                if agent_i < 3:   #simple_tag
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(obs_j[:, 2:]).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
                elif agent_i == 3:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[2], self.mle_base[2]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(obs_j[:, 2:-2]).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
            elif self.nagents == 3:
                if agent_i < 1:     #simple_adv
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:,:2],observations[agent_i]),1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]

                elif agent_i >= 1:
                    self.agents[agent_i].mle = [self.mle_base[2],self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(observations[agent_i]).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
            elif self.nagents == 2:
                if agent_i < 1:     #simple_push
                    self.agents[agent_i].mle = [self.mle_base[0]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(observations[agent_i][:,2:]).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(observations[agent_i][:,2:]).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]

            observations_.append(torch.cat((observations[agent_i], torch.cat(actions, 1)), 1))

        return observations_

    def trian_tag(self, agent_i, KL_criterion, obs, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](obs[0][:, 2:])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](obs[3][:, 2:])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[3].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

            self.mle_opts[2].zero_grad()
            action_i = self.mle_base[2](obs[0][:, 2:-2])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[2])
            torch.nn.utils.clip_grad_norm_(self.mle_base[2].parameters(), 20)
            self.mle_opts[2].step()

    def trian_adv(self, agent_i, KL_criterion, obs, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](torch.cat((obs[1][:,:2],obs[agent_i]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](obs[1])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

            self.mle_opts[2].zero_grad()
            action_i = self.mle_base[2](obs[1])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[2])
            torch.nn.utils.clip_grad_norm_(self.mle_base[2].parameters(), 20)
            self.mle_opts[2].step()

    def trian_push(self, agent_i, KL_criterion, obs, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](obs[0][:, 2:])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](obs[1][:, 2:])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

    def trian_com(self, agent_i, KL_criterion, obs, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](obs[1][:, 4:24])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](obs[4][:, 4:24])
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[4].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()




    def update(self, sample, agent_i, parallel=False, logger=None, sample_r=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        # print('___update___')
        obs, acs, rews, next_obs, dones = sample

        next_obs_ = self._get_obs(next_obs)
        obs_ = self._get_obs(obs)
        curr_agent = self.agents[agent_i]
        # mle
        KL_criterion = torch.nn.KLDivLoss(reduction='sum')
        # for i in range(len(curr_agent.mle)):
        #     curr_agent.mle_optimizer[i].zero_grad()
        #     action_i = curr_agent.mle[i](obs[agent_i]obs[agent_i])
        #     action_pre = gumbel_softmax(action_i, hard=True)
        #     loss = KL_criterion(action_pre.float(), acs[i].float())
        #     loss.backward()
        #     if parallel:
        #         average_gradients(curr_agent.mle[i])
        #     torch.nn.utils.clip_grad_norm_(curr_agent.mle[i].parameters(), 20)
        #     curr_agent.policy_optimizer.step()
        if self.nagents == 6:
            self.trian_com(agent_i, KL_criterion, obs, parallel, acs)
        if self.nagents == 4:
            self.trian_tag(agent_i, KL_criterion, obs, parallel, acs)
        elif self.nagents == 3:
            self.trian_adv(agent_i, KL_criterion, obs, parallel, acs)
        elif self.nagents == 2:
            self.trian_push(agent_i, KL_criterion, obs, parallel, acs)

        # center critic
        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG_ToM':
            all_trgt_acs = []
            if self.discrete_action:  # one-hot encode action

                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs_)]
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)

        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG_ToM':
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

            curr_pol_out = curr_agent.policy(obs_[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG_ToM':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs_):
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
                     'mle_params': [self.get_params()],}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, device, agent_alg="MADDPG_ToM", adversary_alg="MADDPG_ToM",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, output_style='sum'):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
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
            if algtype == "MADDPG_ToM":
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
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'num_in_mle': num_in_mle,})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'device': device,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
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
        for a, params in zip([instance], save_dict['mle_params']):
            a.load_params(params)
        return instance

    def get_params(self):
        params = {
                }
        for i in range(len(self.mle_base)):
            params['mle%d'%i] = self.mle_base[i].state_dict()
            params['mle_optimizer%d'%i] = self.mle_opts[i].state_dict()
        return params

    def load_params(self, params):
        for i in range(len(self.mle_base)):
            self.mle_base[i].load_state_dict(params['mle%d'%i])
            self.mle_opts[i].load_state_dict(params['mle_optimizer%d'%i])

class ToM_SA(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, agent_init_params, alg_types, output_style, device,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.device = device
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent_ToM(lr=lr, discrete_action=discrete_action,
                                     hidden_dim=hidden_dim,
                                     **params, output_style=output_style,
                                     num_agents=self.nagents,
                                     device=self.device)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        if self.nagents == 6:
            self.mle_base = [SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,      #simple_com
                                        self.agent_init_params[3]['num_out_pol'], #adv self-self
                                  hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'],  # adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                             ]
        if self.nagents == 4:
            self.mle_base = [SNNNetwork(self.agent_init_params[0]['num_in_mle'] - 2 + 5,      #simple_tag
                                        self.agent_init_params[0]['num_out_pol'], #adv self-self
                                  hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 2 + 5,
                                        self.agent_init_params[3]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 2 + 5,
                                        self.agent_init_params[3]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                             ]
        elif self.nagents == 3:
            self.mle_base = [SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,      #simple_adv
                                        self.agent_init_params[1]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,
                                        self.agent_init_params[1]['num_out_pol'], #agent self-self
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,
                                        self.agent_init_params[1]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
            ]
        elif self.nagents == 2:
            self.mle_base = [SNNNetwork(self.agent_init_params[0]['num_in_mle']-2  + 5,      #simple_push
                                        self.agent_init_params[0]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle']-2  + 5,
                                        self.agent_init_params[1]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                ]
        self.mle_opts = [Adam(self.mle_base[i].parameters(), lr=lr) for i in range(len(self.mle_base))]
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
        for agent_i, obs in enumerate(observations):
            obs_ = observations_.copy()
            acs_pre_ = actions_pre_.copy()
            obs_.pop(agent_i)
            acs_pre_.pop(agent_i)
            # actions = [self.agents[agent_i].mle[j].cpu()(observations[agent_i]) for j, obs_j in enumerate(obs_)]
            # observations[agent_i] = torch.cat((observations[agent_i], torch.cat(actions, 1)), 1)
            if self.nagents == 6:
                if agent_i < 4:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0],self.mle_base[0], self.mle_base[1], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    # t1 = time.time()
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:, 4:24], acs_pre_[j][:,:5]),1).to(self.device)),
                    #                           hard=True).cpu()
                    #            for j, obs_j in enumerate(obs_)]
                    # print(t1 - time.time())
                    # t1 = time.time()
                    actions = [torch.cat((obs_j[:, 4:24], acs_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)), hard=True).cpu()
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)), hard=True).cpu()
                    actions = torch.cat((b1[:20], b1[20:40], b1[40:60], b2[:20], b2[20:40]), 1)
                    # print(t1 - time.time())
                    # print()
                else:
                    self.agents[agent_i].mle = [self.mle_base[3],self.mle_base[3], self.mle_base[3], self.mle_base[3], self.mle_base[2]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:-2]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:, 4:24], acs_pre_[j]),1).to(self.device)),
                    #                           hard=True).cpu()
                    # for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[1]['num_out_pol']))
                               for j, obs_j in enumerate(obs_)]
                    actions = torch.cat(actions,1)
                    # print()

            if self.nagents == 4:
                if agent_i < 3:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:, 2:], acs_pre_[j]),1).to(self.device)),
                                              hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]

                elif agent_i == 3:
                    self.agents[agent_i].mle = [self.mle_base[2],self.mle_base[2], self.mle_base[2]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:-2]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol']))
                               for j, obs_j in enumerate(obs_)]
            elif self.nagents == 3: #simple_adv
                actions = []
                if agent_i < 1:
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0]]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol']))
                               for j, obs_j in enumerate(obs_)]
                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(0)]), 1).to(self.device)),
                                              hard=True).cpu() )
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(2)]), 1).to(self.device)),
                                              hard=True).cpu() )
                elif agent_i == 2:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(0)]), 1).to(self.device)),
                                              hard=True).cpu() )
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(1)]), 1).to(self.device)),
                                              hard=True).cpu() )

            elif self.nagents == 2:
                if agent_i < 1:
                    self.agents[agent_i].mle = [self.mle_base[0]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i][:,2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0], self.agent_init_params[0]['num_out_pol'])) for j, obs_j in enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i][:, 2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((observations_[agent_i][:,2:],
                                                                     actions_pre[(self.nagents -1 - agent_i)]), 1).to(self.device)),
                                              hard=True).cpu() for j, obs_j in enumerate(obs_)]
            if self.nagents == 6:
                observations[agent_i] = torch.cat((observations[agent_i], actions), 1)
            else:
                observations[agent_i] = torch.cat((observations[agent_i], torch.cat(actions, 1)), 1)
        # t2 = time.time()
        # print('step+time:', t2 - t1)
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]



    def _get_obs(self, observations, actions_pre):
        observations_ = []
        actions_pre_ = []
        for agent_i, obs in enumerate(observations):
            obs_ = observations.copy()
            obs_.pop(agent_i)
            actions_pre_ = actions_pre.copy()
            actions_pre_.pop(agent_i)
            if self.nagents == 6:
                if agent_i < 4:   #simple_comm
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[1], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:, 4:24], actions_pre_[j][:,:5]),1)).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.cat((obs_j[:, 4:24], actions_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)).detach(), hard=True)
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)).detach(), hard=True)
                    actions = torch.cat((b1[:1024], b1[1024:2048], b1[2048:3072], b2[:1024], b2[1024:2048]), 1)

                    # print()
                elif agent_i > 4:
                    self.agents[agent_i].mle = [self.mle_base[3], self.mle_base[3], self.mle_base[3], self.mle_base[3], self.mle_base[2]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(obs_j[:, 4:24]).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0], self.agent_init_params[1]['num_out_pol'])).to(
                        torch.device(self.device)).detach()  for j, obs_j in enumerate(obs_)]
                    actions = torch.cat(actions, 1)
                    # print()
            if self.nagents == 4:
                if agent_i < 3:   #simple_tag
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:, 2:], actions_pre_[j]),1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
                elif agent_i == 3:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[2], self.mle_base[2]]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol'])).to(torch.device(self.device)).detach()
                               for j, obs_j in enumerate(obs_)]
            elif self.nagents == 3:
                actions = []
                if agent_i < 1:     #simple_adv
                    # self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:,:2],observations[agent_i]),1)).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol'])).to(torch.device(self.device)).detach()
                    for j, obs_j in enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[2],self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(observations[agent_i]).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions.append(
                        gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(0)]), 1).to(self.device)).detach(), hard=True))
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(2)]), 1).to(self.device)).detach(), hard=True))
                elif agent_i == 2:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    actions.append(
                        gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(0)]), 1).to(self.device)).detach(), hard=True))
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(1)]), 1).to(self.device)).detach(), hard=True))

            elif self.nagents == 2:
                if agent_i < 1:     #simple_push
                    self.agents[agent_i].mle = [self.mle_base[0]]
                    actions = [torch.zeros((obs_j.shape[0], self.agent_init_params[0]['num_out_pol'])).to(torch.device(self.device)).detach() for j, obs_j in
                     enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((observations[agent_i][:,2:],
                                                                           actions_pre[(self.nagents -1 - agent_i)]), 1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]

            if self.nagents == 6:
                observations_.append(torch.cat((observations[agent_i], actions), 1))
            else:
                observations_.append(torch.cat((observations[agent_i], torch.cat(actions, 1)), 1))

        return observations_

    def trian_tag(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](torch.cat((obs[0][:, 2:], acs_pre[0]),1))#
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[3][:, 2:], acs_pre[3]),1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[3].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

            # self.mle_opts[2].zero_grad()
            # action_i = self.mle_base[2](obs[0][:, 2:-2])
            # action_pre = gumbel_softmax(action_i, hard=True)
            # loss = KL_criterion(action_pre.float(), acs[0].float())
            # loss.backward()
            # if parallel:
            #     average_gradients(self.mle_base[2])
            # torch.nn.utils.clip_grad_norm_(self.mle_base[2].parameters(), 20)
            # self.mle_opts[2].step()

    def trian_adv(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            # self.mle_opts[0].zero_grad()
            # action_i = self.mle_base[0](torch.cat((obs[1][:,:2],obs[agent_i]), 1))
            # action_pre = gumbel_softmax(action_i, hard=True)
            # loss = KL_criterion(action_pre.float(), acs[1].float())
            # loss.backward(retain_graph=True)
            # if parallel:
            #     average_gradients(self.mle_base[0])
            # torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            # self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[1], acs_pre[2]), 1)) #torch.cat((obs[1], acs_pre[2]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

            self.mle_opts[2].zero_grad()
            action_i = self.mle_base[2](torch.cat((obs[1], acs_pre[0]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[2])
            torch.nn.utils.clip_grad_norm_(self.mle_base[2].parameters(), 20)
            self.mle_opts[2].step()

    def trian_push(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            # self.mle_opts[0].zero_grad()
            # action_i = self.mle_base[0](obs[0][:, 2:])  #torch.cat((obs[agent_i][:,2:], actions[(self.nagents -1 - agent_i)]), 1)
            # action_pre = gumbel_softmax(action_i, hard=True)
            # loss = KL_criterion(action_pre.float(), acs[1].float())
            # loss.backward(retain_graph=True)
            # if parallel:
            #     average_gradients(self.mle_base[0])
            # torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            # self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[1][:,2:], acs_pre[(0)]), 1))  #obs[1][:, 2:]
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

    def trian_com(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](torch.cat((obs[1][:, 4:24], acs_pre[(1)]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[4][:, 4:24], acs_pre[(4)]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[4].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

    def update(self, sample, agent_i, parallel=False, logger=None, sample_r=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        # print('___update___')
        acs_pre, obs, acs, rews, next_obs, dones = sample

        next_obs_ = self._get_obs(next_obs, acs)
        obs_ = self._get_obs(obs, acs_pre)
        curr_agent = self.agents[agent_i]
        # mle
        KL_criterion = torch.nn.KLDivLoss(reduction='sum')
        # for i in range(len(curr_agent.mle)):
        #     curr_agent.mle_optimizer[i].zero_grad()
        #     action_i = curr_agent.mle[i](obs[agent_i]obs[agent_i])
        #     action_pre = gumbel_softmax(action_i, hard=True)
        #     loss = KL_criterion(action_pre.float(), acs[i].float())
        #     loss.backward()
        #     if parallel:
        #         average_gradients(curr_agent.mle[i])
        #     torch.nn.utils.clip_grad_norm_(curr_agent.mle[i].parameters(), 20)
        #     curr_agent.policy_optimizer.step()
        if self.nagents == 6:
            self.trian_com(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 4:
            self.trian_tag(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 3:
            self.trian_adv(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 2:
            self.trian_push(agent_i, KL_criterion, obs, acs_pre, parallel, acs)

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

            curr_pol_out = curr_agent.policy(obs_[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs_):
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
                     'mle_params': [self.get_params()],}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, device, agent_alg="ToM_SA", adversary_alg="ToM_SA",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, output_style='sum'):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
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
            if algtype == "ToM_SA":
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
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'num_in_mle': num_in_mle,})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'device': device,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
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
        for a, params in zip([instance], save_dict['mle_params']):
            a.load_params(params)
        return instance

    def get_params(self):
        params = {
                }
        for i in range(len(self.mle_base)):
            params['mle%d'%i] = self.mle_base[i].state_dict()
            params['mle_optimizer%d'%i] = self.mle_opts[i].state_dict()
        return params

    def load_params(self, params):
        for i in range(len(self.mle_base)):
            self.mle_base[i].load_state_dict(params['mle%d'%i])
            self.mle_opts[i].load_state_dict(params['mle_optimizer%d'%i])

class ToM_S(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, agent_init_params, alg_types, output_style, device,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.device = device
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent_ToM(lr=lr, discrete_action=discrete_action,
                                     hidden_dim=hidden_dim,
                                     **params, output_style=output_style,
                                     num_agents=self.nagents,
                                     device=self.device)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        if self.nagents == 6:
            self.mle_base = [SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,      #simple_com
                                        self.agent_init_params[3]['num_out_pol'], #adv self-self
                                  hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'],  # adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                             ]
        if self.nagents == 4:
            self.mle_base = [SNNNetwork(self.agent_init_params[0]['num_in_mle'] - 2 + 5,      #simple_tag
                                        self.agent_init_params[0]['num_out_pol'], #adv self-self
                                  hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 2 + 5,
                                        self.agent_init_params[3]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 2 + 5,
                                        self.agent_init_params[3]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                             ]
        elif self.nagents == 3:
            self.mle_base = [SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,      #simple_adv
                                        self.agent_init_params[1]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,
                                        self.agent_init_params[1]['num_out_pol'], #agent self-self
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,
                                        self.agent_init_params[1]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
            ]
        elif self.nagents == 2:
            self.mle_base = [SNNNetwork(self.agent_init_params[0]['num_in_mle']-2  + 5,      #simple_push
                                        self.agent_init_params[0]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle']-2  + 5,
                                        self.agent_init_params[1]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                ]
        self.mle_opts = [Adam(self.mle_base[i].parameters(), lr=lr) for i in range(len(self.mle_base))]
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
        for agent_i, obs in enumerate(observations):
            obs_ = observations_.copy()
            acs_pre_ = actions_pre_.copy()
            obs_.pop(agent_i)
            acs_pre_.pop(agent_i)
            # actions = [self.agents[agent_i].mle[j].cpu()(observations[agent_i]) for j, obs_j in enumerate(obs_)]
            # observations[agent_i] = torch.cat((observations[agent_i], torch.cat(actions, 1)), 1)
            if self.nagents == 6:
                if agent_i < 4:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0],self.mle_base[0], self.mle_base[1], self.mle_base[1]]

                    actions = [torch.cat((obs_j[:, 4:24], acs_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)), hard=True).cpu()
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)), hard=True).cpu()
                    actions = torch.cat((b1[:20], b1[20:40], b1[40:60], b2[:20], b2[20:40]), 1)
                    # print(t1 - time.time())
                    # print()
                else:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[1]]
                    actions = [torch.cat((obs_j[:, 4:24], acs_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)), hard=True).cpu()
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)), hard=True).cpu()
                    actions = torch.cat((b1[:20], b1[20:40], b1[40:60], b2[:20], b2[20:40]), 1)
                    # actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[1]['num_out_pol']))
                    #            for j, obs_j in enumerate(obs_)]
                    # actions = torch.cat(actions,1)
                    # print()

            if self.nagents == 4:
                if agent_i < 3:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:, 2:], acs_pre_[j]),1).to(self.device)),
                                              hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]

                elif agent_i == 3:
                    self.agents[agent_i].mle = [self.mle_base[2],self.mle_base[2], self.mle_base[2]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:, 2:-2], acs_pre_[j]),1).to(self.device)),
                                              hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:-2]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    # actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol']))
                    #            for j, obs_j in enumerate(obs_)]
            elif self.nagents == 3: #simple_adv
                actions = []
                if agent_i < 1:
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0]]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol']))
                               for j, obs_j in enumerate(obs_)]
                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(0)]), 1).to(self.device)),
                                              hard=True).cpu() )
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(2)]), 1).to(self.device)),
                                              hard=True).cpu() )
                elif agent_i == 2:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(0)]), 1).to(self.device)),
                                              hard=True).cpu() )
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(1)]), 1).to(self.device)),
                                              hard=True).cpu() )

            elif self.nagents == 2:
                if agent_i < 1:
                    self.agents[agent_i].mle = [self.mle_base[0]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i][:,2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0], self.agent_init_params[0]['num_out_pol'])) for j, obs_j in enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i][:, 2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((observations_[agent_i][:,2:],
                                                                     actions_pre[(self.nagents -1 - agent_i)]), 1).to(self.device)),
                                              hard=True).cpu() for j, obs_j in enumerate(obs_)]
            if self.nagents == 6:
                observations[agent_i] = torch.cat((observations[agent_i], actions), 1)
            else:
                observations[agent_i] = torch.cat((observations[agent_i], torch.cat(actions, 1)), 1)
        # t2 = time.time()
        # print('step+time:', t2 - t1)
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]



    def _get_obs(self, observations, actions_pre):
        observations_ = []
        actions_pre_ = []
        for agent_i, obs in enumerate(observations):
            obs_ = observations.copy()
            obs_.pop(agent_i)
            actions_pre_ = actions_pre.copy()
            actions_pre_.pop(agent_i)
            if self.nagents == 6:
                if agent_i < 4:   #simple_comm
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[1], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:, 4:24], actions_pre_[j][:,:5]),1)).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.cat((obs_j[:, 4:24], actions_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)).detach(), hard=True)
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)).detach(), hard=True)
                    actions = torch.cat((b1[:1024], b1[1024:2048], b1[2048:3072], b2[:1024], b2[1024:2048]), 1)

                    # print()
                elif agent_i > 4:
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[1]]
                    actions = [torch.cat((obs_j[:, 4:24], actions_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)).detach(), hard=True)
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)).detach(), hard=True)
                    actions = torch.cat((b1[:1024], b1[1024:2048], b1[2048:3072], b2[:1024], b2[1024:2048]), 1)
                    # print()
            if self.nagents == 4:
                if agent_i < 3:   #simple_tag
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:, 2:], actions_pre_[j]),1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
                elif agent_i == 3:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[2], self.mle_base[2]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:, 2:-2], actions_pre_[j]),1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
            elif self.nagents == 3:
                actions = []
                if agent_i < 1:     #simple_adv
                    # self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:,:2],observations[agent_i]),1)).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol'])).to(torch.device(self.device)).detach()
                    for j, obs_j in enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[2],self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(observations[agent_i]).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions.append(
                        gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(0)]), 1).to(self.device)).detach(), hard=True))
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(2)]), 1).to(self.device)).detach(), hard=True))
                elif agent_i == 2:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    actions.append(
                        gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(0)]), 1).to(self.device)).detach(), hard=True))
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(1)]), 1).to(self.device)).detach(), hard=True))

            elif self.nagents == 2:
                if agent_i < 1:     #simple_push
                    self.agents[agent_i].mle = [self.mle_base[0]]
                    actions = [torch.zeros((obs_j.shape[0], self.agent_init_params[0]['num_out_pol'])).to(torch.device(self.device)).detach() for j, obs_j in
                     enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((observations[agent_i][:,2:],
                                                                           actions_pre[(self.nagents -1 - agent_i)]), 1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]

            if self.nagents == 6:
                observations_.append(torch.cat((observations[agent_i], actions), 1))
            else:
                observations_.append(torch.cat((observations[agent_i], torch.cat(actions, 1)), 1))

        return observations_

    def trian_tag(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](torch.cat((obs[0][:, 2:], acs_pre[0]),1))#
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[3][:, 2:], acs_pre[3]),1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[3].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

            self.mle_opts[2].zero_grad()
            action_i = self.mle_base[2](torch.cat((obs[0][:, 2:-2], acs_pre[0]),1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[2])
            torch.nn.utils.clip_grad_norm_(self.mle_base[2].parameters(), 20)
            self.mle_opts[2].step()

    def trian_adv(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            # self.mle_opts[0].zero_grad()
            # action_i = self.mle_base[0](torch.cat((obs[1][:,:2],obs[agent_i]), 1))
            # action_pre = gumbel_softmax(action_i, hard=True)
            # loss = KL_criterion(action_pre.float(), acs[1].float())
            # loss.backward(retain_graph=True)
            # if parallel:
            #     average_gradients(self.mle_base[0])
            # torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            # self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[1], acs_pre[2]), 1)) #torch.cat((obs[1], acs_pre[2]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

            self.mle_opts[2].zero_grad()
            action_i = self.mle_base[2](torch.cat((obs[1], acs_pre[0]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[2])
            torch.nn.utils.clip_grad_norm_(self.mle_base[2].parameters(), 20)
            self.mle_opts[2].step()

    def trian_push(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            # self.mle_opts[0].zero_grad()
            # action_i = self.mle_base[0](obs[0][:, 2:])  #torch.cat((obs[agent_i][:,2:], actions[(self.nagents -1 - agent_i)]), 1)
            # action_pre = gumbel_softmax(action_i, hard=True)
            # loss = KL_criterion(action_pre.float(), acs[1].float())
            # loss.backward(retain_graph=True)
            # if parallel:
            #     average_gradients(self.mle_base[0])
            # torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            # self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[1][:,2:], acs_pre[(0)]), 1))  #obs[1][:, 2:]
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

    def trian_com(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](torch.cat((obs[1][:, 4:24], acs_pre[(1)]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[4][:, 4:24], acs_pre[(4)]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[4].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()


    def update(self, sample, agent_i, parallel=False, logger=None, sample_r=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        # print('___update___')
        acs_pre, obs, acs, rews, next_obs, dones = sample

        next_obs_ = self._get_obs(next_obs, acs)
        obs_ = self._get_obs(obs, acs_pre)
        curr_agent = self.agents[agent_i]
        # mle
        KL_criterion = torch.nn.KLDivLoss(reduction='sum')
        # for i in range(len(curr_agent.mle)):
        #     curr_agent.mle_optimizer[i].zero_grad()
        #     action_i = curr_agent.mle[i](obs[agent_i]obs[agent_i])
        #     action_pre = gumbel_softmax(action_i, hard=True)
        #     loss = KL_criterion(action_pre.float(), acs[i].float())
        #     loss.backward()
        #     if parallel:
        #         average_gradients(curr_agent.mle[i])
        #     torch.nn.utils.clip_grad_norm_(curr_agent.mle[i].parameters(), 20)
        #     curr_agent.policy_optimizer.step()
        if self.nagents == 6:
            self.trian_com(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 4:
            self.trian_tag(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 3:
            self.trian_adv(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 2:
            self.trian_push(agent_i, KL_criterion, obs, acs_pre, parallel, acs)

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

            curr_pol_out = curr_agent.policy(obs_[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs_):
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
                     'mle_params': [self.get_params()],}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, device, agent_alg="ToM_S", adversary_alg="ToM_S",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, output_style='sum'):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
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
            if algtype == "ToM_S":
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
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'num_in_mle': num_in_mle,})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'device': device,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
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
        for a, params in zip([instance], save_dict['mle_params']):
            a.load_params(params)
        return instance

    def get_params(self):
        params = {
                }
        for i in range(len(self.mle_base)):
            params['mle%d'%i] = self.mle_base[i].state_dict()
            params['mle_optimizer%d'%i] = self.mle_opts[i].state_dict()
        return params

    def load_params(self, params):
        for i in range(len(self.mle_base)):
            self.mle_base[i].load_state_dict(params['mle%d'%i])
            self.mle_opts[i].load_state_dict(params['mle_optimizer%d'%i])

class ToM_self(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, agent_init_params, alg_types, output_style, device,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.device = device
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent_ToM(lr=lr, discrete_action=discrete_action,
                                     hidden_dim=hidden_dim,
                                     **params, output_style=output_style,
                                     num_agents=self.nagents,
                                     device=self.device)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        if self.nagents == 6:
            self.mle_base = [SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,      #simple_com
                                        self.agent_init_params[3]['num_out_pol'], #adv self-self
                                  hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'],  # adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 14 + 5,
                                        self.agent_init_params[3]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                             ]
        if self.nagents == 4:
            self.mle_base = [SNNNetwork(self.agent_init_params[0]['num_in_mle'] - 2 + 5,      #simple_tag
                                        self.agent_init_params[0]['num_out_pol'], #adv self-self
                                  hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 2 + 5,
                                        self.agent_init_params[3]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[3]['num_in_mle'] - 2 + 5,
                                        self.agent_init_params[3]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                             ]
        elif self.nagents == 3:
            self.mle_base = [SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,      #simple_adv
                                        self.agent_init_params[1]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,
                                        self.agent_init_params[1]['num_out_pol'], #agent self-self
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle'] + 5,
                                        self.agent_init_params[1]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
            ]
        elif self.nagents == 2:
            self.mle_base = [SNNNetwork(self.agent_init_params[0]['num_in_mle']-2  + 5,      #simple_push
                                        self.agent_init_params[0]['num_out_pol'], #adv self-other
                                        hidden_dim=hidden_dim, output_style=output_style),
                             SNNNetwork(self.agent_init_params[1]['num_in_mle']-2  + 5,
                                        self.agent_init_params[1]['num_out_pol'],
                                        hidden_dim=hidden_dim, output_style=output_style),    ##agent self-other
                ]
        self.mle_opts = [Adam(self.mle_base[i].parameters(), lr=lr) for i in range(len(self.mle_base))]
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
        for agent_i, obs in enumerate(observations):
            obs_ = observations_.copy()
            acs_pre_ = actions_pre_.copy()
            obs_.pop(agent_i)
            acs_pre_.pop(agent_i)
            # actions = [self.agents[agent_i].mle[j].cpu()(observations[agent_i]) for j, obs_j in enumerate(obs_)]
            # observations[agent_i] = torch.cat((observations[agent_i], torch.cat(actions, 1)), 1)
            if self.nagents == 6:
                if agent_i < 4:
                    self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0],self.mle_base[0], self.mle_base[0], self.mle_base[0]]

                    actions = [torch.cat((obs_j[:, 4:24], acs_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)), hard=True).cpu()
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)), hard=True).cpu()
                    actions = torch.cat((b1[:20], b1[20:40], b1[40:60], b2[:20], b2[20:40]), 1)
                    # print(t1 - time.time())
                    # print()
                else:
                    self.agents[agent_i].mle = [self.mle_base[1],self.mle_base[1], self.mle_base[1], self.mle_base[1], self.mle_base[1]]
                    actions = [torch.cat((obs_j[:, 4:24], acs_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)), hard=True).cpu()
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)), hard=True).cpu()
                    actions = torch.cat((b1[:20], b1[20:40], b1[40:60], b2[:20], b2[20:40]), 1)
                    # actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[1]['num_out_pol']))
                    #            for j, obs_j in enumerate(obs_)]
                    # actions = torch.cat(actions,1)
                    # print()

            if self.nagents == 4:
                if agent_i < 3:
                    self.agents[agent_i].mle = [self.mle_base[1],self.mle_base[1], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:, 2:14], acs_pre_[j]),1).to(self.device)),
                                              hard=True).cpu()
                               for j, obs_j in enumerate(obs_)]

                elif agent_i == 3:
                    self.agents[agent_i].mle = [self.mle_base[2],self.mle_base[2], self.mle_base[2]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:,2:14], acs_pre_[j]),1).to(self.device)),
                                              hard=True).cpu()  for j, obs_j in enumerate(obs_)]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(obs_j[:,2:-2]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    # actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol']))
                    #            for j, obs_j in enumerate(obs_)]
            elif self.nagents == 3: #simple_adv
                actions = []
                if agent_i < 1:
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0]]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol']))
                               for j, obs_j in enumerate(obs_)]
                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(0)]), 1).to(self.device)),
                                              hard=True).cpu() )
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(2)]), 1).to(self.device)),
                                              hard=True).cpu() )
                elif agent_i == 2:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(0)]), 1).to(self.device)),
                                              hard=True).cpu() )
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations_[agent_i],
                                                                     actions_pre[(1)]), 1).to(self.device)),
                                              hard=True).cpu() )
            elif self.nagents == 2:
                if agent_i < 1:
                    self.agents[agent_i].mle = [self.mle_base[0]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i][:,2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0], self.agent_init_params[0]['num_out_pol'])) for j, obs_j in enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].cpu()(observations_[agent_i][:, 2:]), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((observations_[agent_i][:,2:],
                                                                     actions_pre[(self.nagents -1 - agent_i)]), 1).to(self.device)),
                                              hard=True).cpu() for j, obs_j in enumerate(obs_)]
            if self.nagents == 6:
                observations[agent_i] = torch.cat((observations[agent_i], actions), 1)
            else:
                observations[agent_i] = torch.cat((observations[agent_i], torch.cat(actions, 1)), 1)
        # t2 = time.time()
        # print('step+time:', t2 - t1)
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]



    def _get_obs(self, observations, actions_pre):
        observations_ = []
        actions_pre_ = []
        for agent_i, obs in enumerate(observations):
            obs_ = observations.copy()
            obs_.pop(agent_i)
            actions_pre_ = actions_pre.copy()
            actions_pre_.pop(agent_i)
            if self.nagents == 6:
                if agent_i < 4:   #simple_comm
                    self.agents[agent_i].mle = [self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[0], self.mle_base[0]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:, 4:24], actions_pre_[j][:,:5]),1)).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.cat((obs_j[:, 4:24], actions_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)).detach(), hard=True)
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)).detach(), hard=True)
                    actions = torch.cat((b1[:1024], b1[1024:2048], b1[2048:3072], b2[:1024], b2[1024:2048]), 1)

                    # print()
                elif agent_i > 4:
                    self.agents[agent_i].mle = [self.mle_base[1], self.mle_base[1], self.mle_base[1], self.mle_base[1], self.mle_base[1]]
                    actions = [torch.cat((obs_j[:, 4:24], actions_pre_[j][:,:5]),1) for j, obs_j in enumerate(obs_)]
                    b1 = gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat(actions[:3]).to(self.device)).detach(), hard=True)
                    b2 = gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat(actions[3:]).to(self.device)).detach(), hard=True)
                    actions = torch.cat((b1[:1024], b1[1024:2048], b1[2048:3072], b2[:1024], b2[1024:2048]), 1)
                    # print()
            if self.nagents == 4:
                if agent_i < 3:   #simple_tag
                    self.agents[agent_i].mle = [self.mle_base[1], self.mle_base[1], self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:, 2:14], actions_pre_[j]),1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
                elif agent_i == 3:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[2], self.mle_base[2]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:, 2:14], actions_pre_[j]),1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(self.device)(torch.cat((obs_j[:,2:-2], acs_pre_[j]),1).to(self.device)),
                    #                           hard=True).cpu()  for j, obs_j in enumerate(obs_)]
                    # actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol'])).to(torch.device(self.device)).detach()
                    #            for j, obs_j in enumerate(obs_)]
            elif self.nagents == 3:
                actions = []
                if agent_i < 1:     #simple_adv
                    # self.agents[agent_i].mle = [self.mle_base[0],self.mle_base[0]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((obs_j[:,:2],observations[agent_i]),1)).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions = [torch.zeros((obs_j.shape[0],self.agent_init_params[0]['num_out_pol'])).to(torch.device(self.device)).detach()
                    for j, obs_j in enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[2],self.mle_base[1]]
                    # actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(observations[agent_i]).detach(), hard=True)
                    #            for j, obs_j in enumerate(obs_)]
                    actions.append(
                        gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(0)]), 1).to(self.device)).detach(), hard=True))
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(2)]), 1).to(self.device)).detach(), hard=True))
                elif agent_i == 2:
                    self.agents[agent_i].mle = [self.mle_base[2], self.mle_base[1]]
                    actions.append(
                        gumbel_softmax(self.agents[agent_i].mle[0].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(0)]), 1).to(self.device)).detach(), hard=True))
                    actions.append(gumbel_softmax(self.agents[agent_i].mle[1].to(self.device)(torch.cat((observations[agent_i],
                          actions_pre[(1)]), 1).to(self.device)).detach(), hard=True))

            elif self.nagents == 2:
                if agent_i < 1:     #simple_push
                    self.agents[agent_i].mle = [self.mle_base[0]]
                    actions = [torch.zeros((obs_j.shape[0], self.agent_init_params[0]['num_out_pol'])).to(torch.device(self.device)).detach() for j, obs_j in
                     enumerate(obs_)]

                elif agent_i == 1:
                    self.agents[agent_i].mle = [self.mle_base[1]]
                    actions = [gumbel_softmax(self.agents[agent_i].mle[j].to(torch.device(self.device))(torch.cat((observations[agent_i][:,2:],
                                                                           actions_pre[(self.nagents -1 - agent_i)]), 1)).detach(), hard=True)
                               for j, obs_j in enumerate(obs_)]

            if self.nagents == 6:
                observations_.append(torch.cat((observations[agent_i], actions), 1))
            else:
                observations_.append(torch.cat((observations[agent_i], torch.cat(actions, 1)), 1))

        return observations_

    def trian_tag(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[0][:, 2:14], acs_pre[0]),1))#
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

            self.mle_opts[2].zero_grad()
            action_i = self.mle_base[2](torch.cat((obs[3][:, 2:14], acs_pre[3]),1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[3].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[2])
            torch.nn.utils.clip_grad_norm_(self.mle_base[2].parameters(), 20)
            self.mle_opts[2].step()

    def trian_adv(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[1], acs_pre[2]), 1)) #torch.cat((obs[1], acs_pre[2]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

            self.mle_opts[2].zero_grad()
            action_i = self.mle_base[2](torch.cat((obs[1], acs_pre[0]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[2])
            torch.nn.utils.clip_grad_norm_(self.mle_base[2].parameters(), 20)
            self.mle_opts[2].step()

    def trian_push(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[1][:,2:], acs_pre[(0)]), 1))  #obs[1][:, 2:]
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[0].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

    def trian_com(self, agent_i, KL_criterion, obs, acs_pre, parallel, acs):
        if agent_i == 0:
            self.mle_opts[0].zero_grad()
            action_i = self.mle_base[0](torch.cat((obs[1][:, 4:24], acs_pre[(1)]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[1].float())
            loss.backward(retain_graph=True)
            if parallel:
                average_gradients(self.mle_base[0])
            torch.nn.utils.clip_grad_norm_(self.mle_base[0].parameters(), 20)
            self.mle_opts[0].step()

            self.mle_opts[1].zero_grad()
            action_i = self.mle_base[1](torch.cat((obs[4][:, 4:24], acs_pre[(4)]), 1))
            action_pre = gumbel_softmax(action_i, hard=True)
            loss = KL_criterion(action_pre.float(), acs[4].float())
            loss.backward()
            if parallel:
                average_gradients(self.mle_base[1])
            torch.nn.utils.clip_grad_norm_(self.mle_base[1].parameters(), 20)
            self.mle_opts[1].step()

    def update(self, sample, agent_i, parallel=False, logger=None, sample_r=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        # print('___update___')
        acs_pre, obs, acs, rews, next_obs, dones = sample

        next_obs_ = self._get_obs(next_obs, acs)
        obs_ = self._get_obs(obs, acs_pre)
        curr_agent = self.agents[agent_i]
        # mle
        KL_criterion = torch.nn.KLDivLoss(reduction='sum')
        # for i in range(len(curr_agent.mle)):
        #     curr_agent.mle_optimizer[i].zero_grad()
        #     action_i = curr_agent.mle[i](obs[agent_i]obs[agent_i])
        #     action_pre = gumbel_softmax(action_i, hard=True)
        #     loss = KL_criterion(action_pre.float(), acs[i].float())
        #     loss.backward()
        #     if parallel:
        #         average_gradients(curr_agent.mle[i])
        #     torch.nn.utils.clip_grad_norm_(curr_agent.mle[i].parameters(), 20)
        #     curr_agent.policy_optimizer.step()
        if self.nagents == 6:
            self.trian_com(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 4:
            self.trian_tag(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 3:
            self.trian_adv(agent_i, KL_criterion, obs, acs_pre, parallel, acs)
        elif self.nagents == 2:
            self.trian_push(agent_i, KL_criterion, obs, acs_pre, parallel, acs)

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

            curr_pol_out = curr_agent.policy(obs_[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs_):
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
                     'mle_params': [self.get_params()],}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, device, agent_alg="ToM_self", adversary_alg="ToM_self",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, output_style='sum'):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
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
            if algtype == "ToM_self":
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
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'num_in_mle': num_in_mle,})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'device': device,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
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
        for a, params in zip([instance], save_dict['mle_params']):
            a.load_params(params)
        return instance

    def get_params(self):
        params = {
                }
        for i in range(len(self.mle_base)):
            params['mle%d'%i] = self.mle_base[i].state_dict()
            params['mle_optimizer%d'%i] = self.mle_opts[i].state_dict()
        return params

    def load_params(self, params):
        for i in range(len(self.mle_base)):
            self.mle_base[i].load_state_dict(params['mle%d'%i])
            self.mle_opts[i].load_state_dict(params['mle_optimizer%d'%i])

