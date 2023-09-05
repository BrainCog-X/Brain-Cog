import torch
import torch.distributions as td
from utils.networks import MLPNetwork, SNNNetwork, LSTMClassifier
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax


class ToM1(object):
    """
    tom factory (Simplification of ToM's model)
    init ToM0 and ToM1 net
    train ToM0 and ToM1 net
    """

    def __init__(self, tom_base, alg_types, agent_types, num_lm, device, hidden_dim=64):
        self.device = device
        self.alg_types = alg_types
        self.agent_types = agent_types
        self.num_good_agents = len(self._get_index1(self.agent_types, 'agent'))
        self.nagents = len(alg_types)
        self.num_lm = num_lm
        '''
        Assume that ToM0 and ToM1 are equivalent
        '''
        self.tom1 = tom_base
        self.other_tom1 = [0] * self.nagents
        self._agent_tom1_init()
        '''
        ToM0_policy
        '''
        # self.tom_PHI = []   #TODO

        self.hidden = None

    def _agent_tom1_init(self):
        other_alg_types_ = self.alg_types.copy()
        other_agent_types_ = self.agent_types.copy()
        for agent_i in range(self.nagents):
            other_alg_types = other_alg_types_.copy()
            other_agent_types = other_agent_types_.copy()
            other_alg_types.pop(agent_i)
            other_agent_types.pop(agent_i)

            adv_indx = self._get_index1(other_agent_types, 'adversary')
            good_indx = self._get_index1(other_agent_types, 'agent')
            self.other_tom1[agent_i] = [self.tom1['adversary'][self.agent_types[agent_i]]] * len(adv_indx)     #TODO
            self.other_tom1[agent_i] += [self.tom1['agent'][self.agent_types[agent_i]]] * len(good_indx)

    def _get_index1(self, lst=None, item=''):
        return [index for (index, value) in enumerate(lst) if value == item]

    def c_function(self, tom0_actions_q, tom1_actions_q):
        c1 = 0.7

        # tom0_actions = torch.stack([gumbel_softmax(action_i, hard=True)
        #                    for action_i in tom0_actions_prob], 0)
        # tom1_actions = torch.stack([gumbel_softmax(action_i, hard=True)
        #                  for action_i in tom1_actions_prob], 0)
        '''
        batch, num_agent, ep, 1
        '''
        tom0_actions = (tom0_actions_q == tom0_actions_q.max(dim=-1, keepdim=True)[0]).to(dtype=torch.int32)
        tom1_actions = (tom1_actions_q.unsqueeze(1) == tom1_actions_q.unsqueeze(0).max(dim=-1, keepdim=True)[0]).to(
            dtype=torch.int32)
        alig = tom0_actions.long().detach() & tom1_actions.long().detach()
        I_belief = tom0_actions_q * (1 - c1) + alig * c1
        # I_belief = [prob_i * (1 -c1) + alig[i] * c1 for i, prob_i in enumerate(tom0_actions_prob)]

        return I_belief

    def tom1_output(self, agent_i, adv_indx, good_indx, obs_, acs_pre_):
        """
        ToM1 <--> ToM1
        obs_self : obs of self, need to convert
        tom0_out : predict other-action (episode_num * self.args.episode_limit * 2, -1), need to convert
        tom0_out_q : predict other-action q_value (episode_num * self.args.episode_limit * 2, -1)
        device : interact with env (cpu)  train (cuda)

        ToM0_policy
        """
        actions = []
        actions += [
            # gumbel_softmax(
                self.other_tom1[agent_i][j].to(self.device)(
                    torch.cat((obs_[:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                               acs_pre_[:, :5]), 1))#.detach()   #, hard=True
             for j in adv_indx
        ]
        actions += [
            # gumbel_softmax(
                self.other_tom1[agent_i][j].to(self.device)(
                    torch.cat((obs_[:, -(self.num_good_agents * 2 + self.num_lm * 2 + (self.nagents - 1) * 2):],
                               acs_pre_[:, :5]), 1))#.detach()   #, hard=True
             for j in good_indx
        ]
        # E_action = torch.cat(actions, 1)
        E_action = actions
        return E_action











