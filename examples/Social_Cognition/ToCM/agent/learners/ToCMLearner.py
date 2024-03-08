import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from agent.memory.ToCMMemory import ToCMMemory
from agent.models.ToCMModel import ToCMModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout
from agent.optim.utils import advantage
from environments import Env
from networks.ToCM.action import Actor, AttentionActor
from networks.ToCM.critic import MADDPGCritic

torch.autograd.set_detect_anomaly = True
def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class ToCMLearner:  # 通过ToCMLearnerConfig来构建

    def __init__(self, config):
        self.config = config

        self.pretrain_model = False
        self.shared_model = False  # shared pretrain_model
        self.pretrain_actor = False
        self.pretrain_critic = False
        # 根据ToCMLearnerConfig的参数包括：DEVICE, CAPACITY, SEQ_LENGTH, ACTION_SIZE, IN_DIM, FEAT, HIDDEN......
        self.model = ToCMModel(config).to(config.DEVICE).eval()  # wsw TODO 这里已经有了device，为什么挂钩子
        # ToCM Model
        self.actor = Actor(config.IN_DIM+2*(config.num_agents-1), config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(
            config.DEVICE)  # IN_DIM / FEAT  # TODO
        self.critic = MADDPGCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        # 关键点是把model actor critic都放到了device上
        if self.pretrain_model:
            if not self.shared_model:
                self.model.load_state_dict(torch.load(self.load_dir + '28_model.pth'))
            else:
                initialize_weights(self.model, mode='xavier')  # 先全部初始化
                # 加载部分预训练权重
                shared_state_dict = torch.load(self.load_dir + '28_model.pth')
                ignored_layer_keys = ['observation_encoder.fc1.weight', 'observation_decoder.fc2.weight',
                                      'observation_decoder.fc2.bias', 'transition._rnn_input_model.0.weight',
                                      'representation._transition_model._rnn_input_model.0.weight',
                                      'av_action.model.4.weight', 'av_action.model.4.bias', 'q_action.weight',
                                      'q_action.bias']
                for k in ignored_layer_keys:
                    del shared_state_dict[k]
                self.model.load_state_dict(shared_state_dict, strict=False)
            print("Load ToCM Model.")
        else:
            initialize_weights(self.model, mode='xavier')

        if self.pretrain_actor:
            self.actor.load_state_dict(torch.load(self.load_dir + '10_actor.pth'), strict=False)
        else:
            initialize_weights(self.actor, mode='xavier')

        if self.pretrain_critic:
            self.critic.load_state_dict(torch.load(self.load_dir + '10_critic.pth'), strict=False)
        else:
            initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = ToCMMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)
        global wandb
        import wandb
        wandb.init(dir=config.LOG_FOLDER,
                   name=str(config.env_name) + '_' + str(2) +
                        "_seed" + str(config.random_seed) + '131',
                   project=str('mpesnn').upper(),
                   group=str(config.env_name) )  # TODO

    def init_optimizers(self):
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)   # TODO
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer, mode='min', verbose=True)

    def params(self):
        return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])  # 5
        self.total_samples += len(rollout['action'])  # 5
        self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                  rollout['fake'], rollout['last'], rollout.get('avail_action'))
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        if 20000 > self.step_count >= 10000:
            self.config.MODEL_EPOCHS = 10
        if self.step_count >= 20000:
            self.config.MODEL_EPOCHS = 5
        for i in range(self.config.MODEL_EPOCHS):
            samples = self.replay_buffer.sample(self.config.MODEL_BATCH_SIZE)
            self.train_model(samples)

        for i in range(self.config.EPOCHS):
            samples = self.replay_buffer.sample(self.config.BATCH_SIZE)
            # for key, sample in samples.items():
            #     print("key: ", key)
            #     print("sample.shape: ", sample.shape)
            # print("samples.shape: ", samples.shape)
            self.train_agent(samples)

    def train_model(self, samples):  # world model
        # print("Start train")
        self.model.train()
        loss = model_loss(self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
                          samples['reward'], samples['done'], samples['fake'], samples['last'])
        # print("loss: ", loss)
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP, name='model')
        # print("backward by model")
        self.model.eval()

    def train_agent(self, samples):
        actions, av_actions, old_policy, imag_feat, imag_state, obs_pred, returns = actor_rollout(samples['observation'],
                                                                            samples['action'],
                                                                            samples['last'], self.model,
                                                                            self.actor,
                                                                            self.critic if self.config.ENV_TYPE == Env.STARCRAFT  # TODO
                                                                            else self.old_critic,
                                                                            self.config)
        adv = returns.detach() - self.critic(imag_feat, actions).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT or self.config.ENV_TYPE == Env.MPE:
            adv = advantage(adv)  # TODO what adv
        # wandb.log({'Agent/adv': adv.mean()})
        wandb.log({'Agent/Returns': returns.mean()})  # discount algorithm
        # wandb.log({'Agent/Returns max': returns.max()})
        # wandb.log({'Agent/Returns min': returns.min()})
        # wandb.log({'Agent/Returns std': returns.std()})
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):  # 15
                self.cur_update += 1
                idx = inds[i:i + step]
                loss = actor_loss(self.model, imag_state.map(lambda x: x[idx]) ,
                                  obs_pred[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.entropy, self.config)  # TODO
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY, name='actor')
                self.entropy *= self.config.ENTROPY_ANNEALING  # 0.001 0.998
                val_loss = value_loss(self.critic, actions[idx], imag_feat[idx], returns[idx])
                # print("val_loss: ", val_loss)
                if np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY, name='critic')
                # print("backward by agent")
                if self.config.ENV_TYPE == Env.MPE and self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)

    def apply_optimizer(self, opt, model, loss, grad_clip, name=None):  # type of model
        opt.zero_grad()
        loss.backward()  # only here
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 100
        if name is not None and np.random.randint(20) == 9:
            wandb.log({'Grad of '+name: grad_norm})
        opt.step()

    def apply_optimizer_scheduler(self, opt, sch, model, loss, grad_clip, name=None):
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # 100
        # if name is not None:
        #     wandb.log({'Grad of ' + name: grad_norm})
        opt.step()
        sch.step(loss)


