import torch
import os
from network.spiking_net import Critic, VDNNet, Linear_weight, BiasNet
import copy


class SToMVDN:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.loss_trade_off_target = 0
        self.loss_trade_off_eval = 0

        # 神经网络
        self.eval_snn = Critic(input_shape, args)
        self.target_snn = Critic(input_shape, args)
        self.eval_vdn_snn = VDNNet()  # 把agentsQ值加起来的网络
        self.target_vdn_snn = VDNNet()
        self.bias_net = BiasNet(args)
        self.trade_off_net = Linear_weight(2, 1, args)

        self.args = args
        if self.args.cuda:
            self.eval_snn.cuda(self.args.device)
            self.target_snn.cuda(self.args.device)
            self.eval_vdn_snn.cuda(self.args.device)
            self.target_vdn_snn.cuda(self.args.device)
            self.trade_off_net.cuda(self.args.device)
            self.bias_net.cuda(self.args.device)

        self.model_dir = args.model_dir + '/' + args.alg + args.exp_dir + args.save_model_dir
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir):
                # path_snn = '/home/zhaozhuoya/exp2/ToM2_test/model/siql/199_snn_net_params.pkl'
                map_location = self.args.device if self.args.cuda else 'cpu'
                self.eval_snn.load_state_dict(torch.load(self.model_dir, map_location=map_location))
                print('Successfully load the model: {}'.format(self.model_dir))
            else:
                print(self.model_dir)
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_snn.load_state_dict(self.eval_snn.state_dict())
        self.target_vdn_snn.load_state_dict(self.eval_vdn_snn.state_dict())

        self.eval_parameters = list(self.eval_snn.parameters()) + \
                               list(self.eval_vdn_snn.parameters()) + \
                                    list(self.trade_off_net.parameters()) + \
                                    list(self.bias_net.parameters())

        self.trade_off_parameters = list(self.trade_off_net.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)
            self.optimizer_T = torch.optim.RMSprop(self.trade_off_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_h1_mem, self.eval_h1_spike = None, None
        self.target_h1_mem, self.target_h1_spike = None, None
        self.eval_h2_mem, self.eval_h2_spike = None, None
        self.target_h2_mem, self.target_h2_spike = None, None
        print('Init alg SCOVDN_ToM')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        '''
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        '''
        episode_num = batch['O'].shape[0]
        self.init_hidden_learn(episode_num)
        self.bias_net.reset(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == 'U':  # 'O'
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, avail_u_next, terminated = batch['U'], batch['R'].squeeze(-1), batch['AVAIL_U'], \
                                                  batch['AVAIL_U_NEXT'], batch['TERMINATE'].repeat(1, 1, self.n_agents)
        mask = (1 - batch["PADDED"].float()).repeat(1, 1, self.n_agents)  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        if self.args.cuda:
            u = u.cuda(self.args.device)
            r = r.cuda(self.args.device)
            terminated = terminated.cuda(self.args.device)
            mask = mask.cuda(self.args.device)
            # self.bias_net.cuda(self.args.device)
        u = u.to(torch.int64)

        # ---------------------------------------independent_Q_net------------------------------------------------------
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        q_evals, q_targets, hidden_evals, hidden_targets = self.get_q_values(batch, max_episode_len)
        # ---------------------------------------independent_Q_net------------------------------------------------------

        # --------------------------------------------bias_net----------------------------------------------------------
        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len， n_agents， n_actions)
        v = self.get_bias(batch, hidden_evals, hidden_targets, episode_num)
        # --------------------------------------------bias_net----------------------------------------------------------


        # ---------------------------------------_self+Q_other_net------------------------------------------------------
        q_other_evals, q_other_targets = q_evals[:, :, [1, 0], :].unsqueeze(4), q_targets[:, :, [1, 0], :].unsqueeze(4)
        q_evals_, q_targets_ = q_evals.unsqueeze(4), q_targets.unsqueeze(4)
        q_total_evals = torch.cat((q_evals_, q_other_evals), 4)
        q_total_targets = torch.cat((q_targets_, q_other_targets), 4)
        # q_total_evals = self.trade_off_net(q_total_evals)   #([10, 50, 2, 5, 1])
        # q_total_targets = self.trade_off_net(q_total_targets)  # ([10, 50, 2, 5, 1])
        q_total_evals = q_evals_ + q_other_targets
        q_total_targets = q_targets_ + q_other_targets
        # --------------------------------------------_self+Q_other_net-------------------------------------------------

        # --------------------------------------------L_self/other------------------------------------------------------
        q_total_targets[avail_u_next == 0.0] = - 9999999
        q_total_targets = q_total_targets.max(dim=3)[0].squeeze()

        q_total_evals = torch.gather(q_total_evals.squeeze(4), dim=3, index=u).squeeze(3)

        y = r + self.args.gamma * q_total_targets * (1 - terminated)
        td_error = q_total_evals - y.detach()
        l_so = ((td_error * mask) ** 2).sum() / mask.sum()
        # --------------------------------------------L_self/other------------------------------------------------------

        # --------------------------------------------action_prob_Q-----------------------------------------------------
        # probablity of action
        action_prob = self._get_action_prob(batch, max_episode_len, 0.4)  # 每个agent的所有动作的概率self.args.epsilon
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)  # 每个agent的选择的动作对应的概率
        pi_taken[mask == 0] = 1.0  # 因为要取对数，对于那些填充的经验，所有概率都为0，取了log就是负无穷了，所以让它们变成1
        log_pi_taken = torch.log(pi_taken)
        # --------------------------------------------action_prob_Q-----------------------------------------------------

        # ----------------------------------------------L_coma----------------------------------------------------------
        # q_evals = torch.gather(q_evals * action_prob, dim=3, index=u).squeeze(3)
        q_evals_coma = (q_evals * action_prob).sum(dim=3, keepdim=True).squeeze(3)
        coma_error = q_evals_coma.sum(dim=-1) - q_total_targets.detach().sum(dim=-1) + v
        l_coma = ((coma_error * mask[:,:,0]) ** 2).sum() / mask[:,:,0].sum()
        # ----------------------------------------------L_coma----------------------------------------------------------

        # -----------------------------------------------L_sum----------------------------------------------------------
        q_evals_sum = self.eval_vdn_snn(q_evals)
        sum_error = q_evals_sum.sum(dim=-1).squeeze(2) - q_total_targets.detach().sum(dim=-1) + v
        l_sum = ((sum_error * mask[:,:,0]) ** 2).sum() / mask[:,:,0].sum()
        # -----------------------------------------------L_sum----------------------------------------------------------

        LOSS = l_so + l_coma + l_sum

        self.optimizer.zero_grad()
        LOSS.backward()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_snn.load_state_dict(self.eval_snn.state_dict())
            self.target_vdn_snn.load_state_dict(self.eval_vdn_snn.state_dict())
        return LOSS

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = batch['O'][:, transition_idx], \
                                  batch['O_NEXT'][:, transition_idx], batch['U_ONEHOT'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['O'].shape[0]
        q_evals, q_targets, eval_h2_mems, target_h2_mems = [], [], [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda(self.args.device)
                inputs_next = inputs_next.cuda(self.args.device)
                self.eval_h1_mem, self.eval_h1_spike, self.eval_h2_mem, self.eval_h2_spike = \
                    self.eval_h1_mem.cuda(self.args.device), self.eval_h1_spike.cuda(
                        self.args.device), self.eval_h2_mem.cuda(self.args.device), self.eval_h2_spike.cuda(
                        self.args.device)
                self.target_h1_mem, self.target_h1_spike, self.target_h2_mem, self.target_h2_spike = \
                    self.target_h1_mem.cuda(self.args.device), self.target_h1_spike.cuda(
                        self.args.device), self.target_h2_mem.cuda(self.args.device), self.target_h2_spike.cuda(
                        self.args.device)
            q_eval, self.eval_h1_mem, self.eval_h1_spike, self.eval_h2_mem, self.eval_h2_spike = \
                self.eval_snn(inputs, self.eval_h1_mem, self.eval_h1_spike, self.eval_h2_mem,
                              self.eval_h2_spike)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_h1_mem, self.target_h1_spike, self.target_h2_mem, self.target_h2_spike = \
                self.target_snn(inputs_next, self.target_h1_mem, self.target_h1_spike, self.target_h2_mem,
                                self.target_h2_spike)

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            eval_h2_mem = self.eval_h2_mem.view(episode_num, self.n_agents, -1)
            target_h2_mem = self.target_h2_mem.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
            eval_h2_mems.append(eval_h2_mem)
            target_h2_mems.append(target_h2_mem)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        hidden_evals = torch.stack(eval_h2_mems, dim=1)
        hidden_targets = torch.stack(target_h2_mems, dim=1)
        return q_evals, q_targets, hidden_evals, hidden_targets

    def get_bias(self, batch, hidden_evals, hidden_targets, episode_num, hat=False):
        # episode_num, max_episode_len, _, _ = hidden_targets.shape
        max_episode_len = self.args.episode_limit
        states = batch['O'][:, :max_episode_len]
        states_next = batch['O_NEXT'][:, :max_episode_len]
        u_onehot = batch['U_ONEHOT'][:, :max_episode_len]
        if self.args.cuda:
            states = states.cuda(self.args.device)[:,:,0,:]
            states_next = states_next.cuda(self.args.device)[:,:,0,:]
            u_onehot = u_onehot.cuda(self.args.device)
            hidden_evals = hidden_evals.cuda(self.args.device)
            hidden_targets = hidden_targets.cuda(self.args.device)
        if hat:
            v = None
        else:
            v = self.bias_net(states, hidden_evals)
            # 把q_eval、q_target、v维度变回(episode_num, max_episode_len)
            v = v.view(episode_num, -1, 1).squeeze(-1)
        return v


    def _get_actor_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, u_onehot = batch['O'][:, transition_idx], batch['U_ONEHOT'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)
        # 给inputs添加上一个动作、agent编号

        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            # 因为当前的inputs三维的数据，每一维分别代表(episode编号，agent编号，inputs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        # 要把inputs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch['O'].shape[0]
        avail_actions = batch['AVAIL_U']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda(self.args.device)
                # self.eval_hidden = self.eval_hidden.cuda(self.args.device)
                self.eval_h1_mem, self.eval_h1_spike, self.eval_h2_mem, self.eval_h2_spike = \
                    self.eval_h1_mem.cuda(self.args.device), self.eval_h1_spike.cuda(
                        self.args.device), self.eval_h2_mem.cuda(self.args.device), self.eval_h2_spike.cuda(
                        self.args.device)
                self.target_h1_mem, self.target_h1_spike, self.target_h2_mem, self.target_h2_spike = \
                    self.target_h1_mem.cuda(self.args.device), self.target_h1_spike.cuda(
                        self.args.device), self.target_h2_mem.cuda(self.args.device), self.target_h2_spike.cuda(
                        self.args.device)

            # outputs, self.eval_hidden = self.eval_snn(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            outputs, self.eval_h1_mem, self.eval_h1_spike, self.eval_h2_mem, self.eval_h2_spike = \
                self.eval_snn(inputs, self.eval_h1_mem, self.eval_h1_spike, self.eval_h2_mem,
                              self.eval_h2_spike)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)

            # 把q_eval维度重新变回(8, 5,n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        # 得的action_prob是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        action_prob = torch.stack(action_prob, dim=1).cpu()

        action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1,
                                                                            avail_actions.shape[-1])  # 可以选择的动作的个数
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        action_prob[avail_actions == 0] = 0.0
        if self.args.cuda:
            action_prob = action_prob.cuda(self.args.device)
        return action_prob

    def init_hidden(self, episode_num, num_env):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_h1_mem = self.eval_h1_spike = torch.zeros(episode_num, self.n_agents, num_env,
                                                            self.args.rnn_hidden_dim)
        self.target_h1_mem = self.target_h1_spike = torch.zeros(episode_num, self.n_agents, num_env,
                                                                self.args.rnn_hidden_dim)
        self.eval_h2_mem = self.eval_h2_spike = torch.zeros(episode_num, self.n_agents, num_env,
                                                            self.args.rnn_hidden_dim)
        self.target_h2_mem = self.target_h2_spike = torch.zeros(episode_num, self.n_agents, num_env,
                                                                self.args.rnn_hidden_dim)

    def init_hidden_learn(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_h1_mem = self.eval_h1_spike = torch.zeros(episode_num, self.n_agents,
                                                            self.args.rnn_hidden_dim)
        self.target_h1_mem = self.target_h1_spike = torch.zeros(episode_num, self.n_agents,
                                                                self.args.rnn_hidden_dim)
        self.eval_h2_mem = self.eval_h2_spike = torch.zeros(episode_num, self.n_agents,
                                                            self.args.rnn_hidden_dim)
        self.target_h2_mem = self.target_h2_spike = torch.zeros(episode_num, self.n_agents,
                                                                self.args.rnn_hidden_dim)

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_snn.state_dict(),
                   self.model_dir + '/' + num + '_snn_net_params_{}.pkl'.format(self.args.num_run))

    def load_model(self, train_step):
        num = str(train_step // self.args.save_cycle)

        path = torch.load(self.model_dir + '/' + num + '_snn_net_params.pkl'.format(self.args.num_run))

        self.eval_snn.load_state_dict(path)

