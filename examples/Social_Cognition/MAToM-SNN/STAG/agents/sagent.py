import numpy as np
import torch
from torch.distributions import Categorical
from braincog.base.encoder.population_coding import PEncoder
from spikingjelly.activation_based import functional
TIMESTEPS = 15
M = 5

# Agent
class Agents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        # encoder
        self.pencoder = PEncoder(TIMESTEPS, 'population_voltage')

        if args.alg == 'ppo':
            from policy.ppo import PPO
            self.policy = PPO(args)
        if args.alg == 'iql':
            from policy.iql import IQL
            self.policy = IQL(args)
            if args.mode == 'test':
                self.policy.load_model(395000)
        if args.alg == 'svdn':
            from policy.svdn import SVDN
            self.policy = SVDN(args)
        if args.alg == 'scovdn':
            from policy.scovdn import SCOVDN
            self.policy = SCOVDN(args)
        if args.alg == 'stomvdn':
            from policy.stomvdn import SToMVDN
            self.policy = SToMVDN(args)
        if args.alg == 'scovdn_weight':
            from policy.scovdn_weight import SCOVDN_W
            self.policy = SCOVDN_W(args)
        if args.alg == 'siql':
            from policy.siql import SIQL
            self.policy = SIQL(args)
        if args.alg == 'scoiql':
            from policy.scoiql import SCOIQL
            self.policy = SCOIQL(args)
        if args.alg == 'siql_e':
            from policy.siql_encoder import SIQL_E
            self.policy = SIQL_E(args)
        if args.alg == 'siql_e2':
            from policy.siql_encoder2 import SIQL_EE
            self.policy = SIQL_EE(args)
        if args.alg == 'siql_no_rnn':
            from policy.siql_no_rnn import SIQLUR
            self.policy = SIQLUR(args)
        if args.alg == 'siql_no_rnn2':
            from policy.siql_no_rnn2 import SIQLUR2
            self.policy = SIQLUR2(args)
        self.args = args

    def choose_action(self, num_env, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros((num_env, self.n_agents))
        agent_id[:, agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)  # torch.Size([1, 17])
        # init hidden tensor
        if self.args.alg == 'siql_e' or self.args.alg == 'siql_e2':
            h1_mem = self.policy.eval_h1_mem[:, agent_num, :, :, :, :]
            h1_spike = self.policy.eval_h1_spike[:, agent_num, :, :, :, :]
            h2_mem = self.policy.eval_h2_mem[:, agent_num, :, :, :, :]
            h2_spike = self.policy.eval_h2_spike[:, agent_num, :, :, :, :]
            inputs_, _ = self.pencoder(inputs=inputs, num_popneurons=M, VTH=0.99)    ###########################################################
            inputs = torch.transpose(inputs_, 0, 3)
            inputs = inputs.squeeze().unsqueeze(0)

        else:

            h1_mem = self.policy.eval_h1_mem[:, agent_num, :, :]    #
            h1_spike = self.policy.eval_h1_spike[:, agent_num, :, :]
            h2_mem = self.policy.eval_h2_mem[:, agent_num, :, :]
            h2_spike = self.policy.eval_h2_spike[:, agent_num, :, :]


        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda(self.args.device)
            h1_mem = h1_mem.cuda(self.args.device)
            h1_spike = h1_spike.cuda(self.args.device)
            h2_mem = h2_mem.cuda(self.args.device)
            h2_spike = h2_spike.cuda(self.args.device)
        # get q value
        if self.args.alg == 'siql_no_rnn' or self.args.alg == 'siql_no_rnn2':
            self.policy.eval_snn.reset()
            q_value = self.policy.eval_snn(inputs)
            # functional.reset_net(self.policy_sc.eval_snn)
        else:
            q_value, self.policy.eval_h1_mem[:, agent_num, :], self.policy.eval_h1_spike[:, agent_num, :],\
                self.policy.eval_h2_mem[:, agent_num, :], self.policy.eval_h2_spike[:, agent_num, :]= \
                self.policy.eval_snn(inputs, h1_mem, h1_spike, h2_mem, h2_spike)

        # choose action from q value
        # q_value[avail_actions == 0.0] = - float("inf")
        if self.args.alg == 'siql_e' or self.args.alg == 'siql_e2':
            q_value = q_value.sum(dim=2)
            q_value = q_value.sum(dim=2)
        if np.random.uniform() < epsilon:
            # action = np.random.choice(avail_actions_ind)  # action是一个整数
            action = torch.tensor([[np.random.choice(avail_actions_ind) for i in range(num_env)]])
        else:
            action = torch.argmax(q_value, 2)
        return action


    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param_sc inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['TERMINATE']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)



