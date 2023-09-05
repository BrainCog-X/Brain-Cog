import torch
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork, RNN, SNNNetwork, LSTMClassifier
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import time

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = LSTMClassifier(num_in_pol, num_out_pol,#MLPNetwork
                                 hidden_dim,)
                                 # constrain_out=True,
                                 # discrete_action=discrete_action)
        self.critic = LSTMClassifier(num_in_critic, 1,
                                 hidden_dim,)
                                 # constrain_out=False)
        self.target_policy = LSTMClassifier(num_in_pol, num_out_pol,
                                        hidden_dim,)
                                        # constrain_out=True,
                                        # discrete_action=discrete_action)
        self.target_critic = LSTMClassifier(num_in_critic, 1,
                                        hidden_dim,)
                                        # constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)


        if self.discrete_action:
            if explore:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (gumbel_softmax(action[:, :5], hard=True), gumbel_softmax(action[:, 5:], hard=True)), 1)
                else:
                    action = gumbel_softmax(action, hard=True)
            else:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (onehot_from_logits(action[:, :5]), onehot_from_logits(action[:, 5:])), 1)
                else:
                    action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class DDPGAgent_RNN(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = RNN(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = RNN(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = RNN(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = RNN(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)

        self.policy_hidden = None
        self.policy_target_hidden = None
        self.critic_hidden = None
        self.critic_target_hidden = None
        self.num_in_pol = num_in_pol
        self.num_out_pol = num_out_pol
        self.hidden_dim = hidden_dim
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        action, self.policy_hidden = self.policy(obs, self.policy_hidden)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

    def init_hidden(self, len_ep, policy_hidden=False, policy_target_hidden=False, \
                    critic_hidden=False, critic_target_hidden=False):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        if policy_hidden == True:
            self.policy_hidden = torch.zeros((len_ep, self.hidden_dim))
        if policy_target_hidden == True:
            self.policy_target_hidden = torch.zeros((len_ep, self.hidden_dim))
        if critic_hidden == True:
            self.critic_hidden = torch.zeros((len_ep, self.hidden_dim))
        if critic_target_hidden == True:
            self.critic_target_hidden = torch.zeros((len_ep, self.hidden_dim))

class DDPGAgent_SNN(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, output_style, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = SNNNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 output_style=output_style)
        self.critic = SNNNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 output_style=output_style)
        self.target_policy = SNNNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        output_style=output_style)
        self.target_critic = SNNNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        output_style=output_style)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        # t1 = time.time()
        action = self.policy(obs)
        # t2 = time.time()
        # print('time_interaction:', t2 - t1)
        if self.discrete_action:
            if explore:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (gumbel_softmax(action[:, :5], hard=True), gumbel_softmax(action[:, 5:], hard=True)), 1)
                else:
                    action = gumbel_softmax(action, hard=True)
            else:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (onehot_from_logits(action[:, :5]), onehot_from_logits(action[:, 5:])), 1)
                else:
                    action = onehot_from_logits(action)
            # if explore:
            #
            #     action = gumbel_softmax(action, hard=True)
            #
            # else:
            #     action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)

            action = action.clamp(-1, 1)

        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class DDPGAgent_ToM(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_in_mle, output_style,
                 num_agents, device, hidden_dim=64, lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.device = device
        self.policy = LSTMClassifier(num_in_pol, num_out_pol,hidden_dim) #SNNNetwork
                                 # hidden_dim=hidden_dim,
                                 # output_style=output_style)
        self.critic = LSTMClassifier(num_in_critic, 1,hidden_dim)
                                 # hidden_dim=hidden_dim,
                                 # output_style=output_style)
        self.target_policy = LSTMClassifier(num_in_pol, num_out_pol,hidden_dim)
                                        # hidden_dim=hidden_dim,
                                        # output_style=output_style)
        self.target_critic = LSTMClassifier(num_in_critic, 1,hidden_dim)
                                        # hidden_dim=hidden_dim,
                                        # output_style=output_style)
        # self.policy = SNNNetwork(num_in_pol, num_out_pol,
        #                          hidden_dim=hidden_dim,
        #                          output_style=output_style)
        # self.critic = SNNNetwork(num_in_critic, 1,
        #                          hidden_dim=hidden_dim,
        #                          output_style=output_style)
        # self.target_policy = SNNNetwork(num_in_pol, num_out_pol,
        #                                 hidden_dim=hidden_dim,
        #                                 output_style=output_style)
        # self.target_critic = SNNNetwork(num_in_critic, 1,
        #                                 hidden_dim=hidden_dim,
        #                                 output_style=output_style)
        # self.mle = [SNNNetwork(num_in_mle, num_out_pol,
        #                       hidden_dim=hidden_dim,
        #                       output_style=output_style)] * (num_agents - 1)
        self.mle = []
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.mle_optimizer = []
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy.to(self.device)(obs.to(self.device))
        if self.discrete_action:
            if explore:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (gumbel_softmax(action[:, :5], hard=True), gumbel_softmax(action[:, 5:], hard=True)), 1).cpu()
                else:
                    action = gumbel_softmax(action, hard=True).cpu()
            else:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (onehot_from_logits(action[:, :5], hard=True), onehot_from_logits(action[:, 5:], hard=True)), 1)
                else:
                    action = onehot_from_logits(action).cpu()
            # if explore:
            #     action = gumbel_softmax(action, hard=True).cpu()
            # else:
            #     action = onehot_from_logits(action).cpu()
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)

        return action

    def get_params(self):
        params = {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                }
        # for i in range(len(self.mle)):
        #     params['mle%d'%i] = self.mle[i].state_dict()
        #     params['mle_optimizer%d'%i] = self.mle_optimizer[i].state_dict()
        return params

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        # for i in range(len(self.mle)):
        #     self.mle[i].load_state_dict(params['mle%d'%i])
        #     self.mle_optimizer[i].load_state_dict(params['mle_optimizer%d'%i])

class rDDPGAgent_ToM(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, num_in_mle, output_style,
                 num_agents, device, hidden_dim=64, lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.device = device
        self.policy = RNN(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                          constrain_out=True,
                          discrete_action=discrete_action)
        self.critic = RNN(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.target_policy = RNN(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.target_critic = RNN(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        # self.mle = [SNNNetwork(num_in_mle, num_out_pol,
        #                       hidden_dim=hidden_dim,
        #                       output_style=output_style)] * (num_agents - 1)
        self.mle = []
        self.policy_hidden = None
        self.policy_target_hidden = None
        self.critic_hidden = None
        self.critic_target_hidden = None
        self.hidden_dim = hidden_dim
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.mle_optimizer = []
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action, self.policy_hidden = self.policy(obs, self.policy_hidden)
        if self.discrete_action:
            if explore:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (gumbel_softmax(action[:, :5], hard=True), gumbel_softmax(action[:, 5:], hard=True)), 1).cpu()
                else:
                    action = gumbel_softmax(action, hard=True).cpu()
            else:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (onehot_from_logits(action[:, :5], hard=True), onehot_from_logits(action[:, 5:], hard=True)), 1)
                else:
                    action = onehot_from_logits(action).cpu()
            # if explore:
            #     action = gumbel_softmax(action, hard=True).cpu()
            # else:
            #     action = onehot_from_logits(action).cpu()
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)

        return action

    def get_params(self):
        params = {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                }
        # for i in range(len(self.mle)):
        #     params['mle%d'%i] = self.mle[i].state_dict()
        #     params['mle_optimizer%d'%i] = self.mle_optimizer[i].state_dict()
        return params

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        # for i in range(len(self.mle)):
        #     self.mle[i].load_state_dict(params['mle%d'%i])
        #     self.mle_optimizer[i].load_state_dict(params['mle_optimizer%d'%i])

    def init_hidden(self, len_ep, policy_hidden=False, policy_target_hidden=False, \
                    critic_hidden=False, critic_target_hidden=False):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        if policy_hidden == True:
            self.policy_hidden = torch.zeros((len_ep, self.hidden_dim))
        if policy_target_hidden == True:
            self.policy_target_hidden = torch.zeros((len_ep, self.hidden_dim))
        if critic_hidden == True:
            self.critic_hidden = torch.zeros((len_ep, self.hidden_dim))
        if critic_target_hidden == True:
            self.critic_target_hidden = torch.zeros((len_ep, self.hidden_dim))

class lDDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = LSTMClassifier(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = LSTMClassifier(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = LSTMClassifier(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = LSTMClassifier(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)


        if self.discrete_action:
            if explore:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (gumbel_softmax(action[:, :5], hard=True), gumbel_softmax(action[:, 5:], hard=True)), 1)
                else:
                    action = gumbel_softmax(action, hard=True)
            else:
                if action.shape[1] == 9:
                    action = torch.cat(
                        (onehot_from_logits(action[:, :5]), onehot_from_logits(action[:, 5:])), 1)
                else:
                    action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])