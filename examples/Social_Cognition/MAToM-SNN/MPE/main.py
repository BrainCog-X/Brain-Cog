import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer, ReplayBuffer_pre
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from policy.maddpg import MADDPG, MADDPG_SNN, MADDPG_ToM, ToM_SA, ToM_S, ToM_self
from tqdm import tqdm

def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='simple_world_comm', type=str,
                        choices=['simple_tag', 'simple_adversary', 'simple_push', 'simple_world_comm'],
                        help="Name of environment")
    parser.add_argument("--model_name", default='ann', type=str,
                        help="Name of directory to store " +
                             "model/training contents") #ToM_SA
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--cuda_num",
                        default=7, type=int,
                        help="device")
    parser.add_argument("--output_style",
                        default='sum', type=str,
                        choices=['sum', 'voltage'])
    parser.add_argument("--n_rollout_threads", default=20, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=15000, type=int)#
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,#4
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG_ToM", type=str,
                        choices=['MADDPG', 'DDPG', 'MADDPG_SNN', 'MADDPG_ToM', 'ToM_SA', 'ToM_S', 'ToM_self'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG_ToM", type=str,
                        choices=['MADDPG', 'DDPG', 'MADDPG_SNN', 'MADDPG_ToM', 'ToM_SA', 'ToM_S', 'ToM_self'])
    parser.add_argument("--discrete_action",
                        # default=False, type=bool,
                        action='store_true')
    args = parser.parse_args()
    parser.add_argument('--device', type=str, default='cuda:{}'.format(args.cuda_num), help='whether to use the GPU')  #'cuda:1'
    parser = parser.parse_args()
    return parser

USE_CUDA = torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    pbar = tqdm(config.n_episodes)
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    if config.agent_alg == 'MADDPG' or config.agent_alg == 'DDPG':
        print('_____MADDPG_____')
        maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                      adversary_alg=config.adversary_alg,
                                      tau=config.tau,
                                      lr=config.lr,
                                      hidden_dim=config.hidden_dim,
                                      device=config.device)
    elif config.agent_alg == 'MADDPG_SNN':
        print('_____MADDPG_SNN_____')
        maddpg = MADDPG_SNN.init_from_env(env, agent_alg=config.agent_alg,
                                      adversary_alg=config.adversary_alg,
                                      tau=config.tau,
                                      lr=config.lr,
                                      hidden_dim=config.hidden_dim,
                                      output_style=config.output_style,
                                          device=config.device)
    elif config.agent_alg == 'MADDPG_ToM':
        print('_____MADDPG_ToM_____')
        maddpg = MADDPG_ToM.init_from_env(env, agent_alg=config.agent_alg,
                                      adversary_alg=config.adversary_alg,
                                      tau=config.tau,
                                      lr=config.lr,
                                      hidden_dim=config.hidden_dim,
                                      output_style=config.output_style,
                                      device=config.device)

    elif config.agent_alg == 'ToM_SA':
        print('_______ToM_SA_______')
        maddpg = ToM_SA.init_from_env(env, agent_alg=config.agent_alg,
                                      adversary_alg=config.adversary_alg,
                                      tau=config.tau,
                                      lr=config.lr,
                                      hidden_dim=config.hidden_dim,
                                      output_style=config.output_style,
                                      device=config.device)
    elif config.agent_alg == 'ToM_S':
        print('_______ToM_S_______')
        maddpg = ToM_S.init_from_env(env, agent_alg=config.agent_alg,
                                      adversary_alg=config.adversary_alg,
                                      tau=config.tau,
                                      lr=config.lr,
                                      hidden_dim=config.hidden_dim,
                                      output_style=config.output_style,
                                      device=config.device)


        print('_______ToM_self_______')
        maddpg = ToM_self.init_from_env(env, agent_alg=config.agent_alg,
                                      adversary_alg=config.adversary_alg,
                                      tau=config.tau,
                                      lr=config.lr,
                                      hidden_dim=config.hidden_dim,
                                      output_style=config.output_style,
                                      device=config.device)


    if config.agent_alg == 'ToM_SA'or config.agent_alg == 'ToM_S' or config.agent_alg == 'ToM_self' or config.agent_alg == 'ToM_SB':
        replay_buffer = ReplayBuffer_pre(config.buffer_length, maddpg.nagents,
                                     [obsp.shape[0] for obsp in env.observation_space],
                                     [acsp.n if isinstance(acsp, Discrete) else sum(acsp.high - acsp.low + 1)
                                      for acsp in env.action_space],
                                 device=config.device)
    else:

        replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                     [obsp.shape[0] for obsp in env.observation_space],
                                     [acsp.n if isinstance(acsp, Discrete) else sum(acsp.high - acsp.low + 1)
                                      for acsp in env.action_space],
                                 device=config.device)

    t = 0
    total_reward = []
    for agent_i in range(maddpg.nagents):
        total_reward.append([])
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        # print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                 ep_i + 1 + config.n_rollout_threads,
        #                                 config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')
        torch_agent_actions = [torch.zeros((config.n_rollout_threads, 5)) for i in range(maddpg.nagents)]
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()
        obs_ep = []
        agent_actions_ep = []
        rewards_ep = []
        next_obs_ep = []
        dones_ep = []
        for et_i in range(config.episode_length):
            torch_agent_actions_pre = torch_agent_actions
            torch_agent_actions_pre = [ac.data.numpy() for ac in torch_agent_actions_pre]
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]    #
            # get actions as torch Variables
            # t1 = time.time()
            if config.agent_alg == 'ToM_SA' or config.agent_alg == 'ToM_S' or config.agent_alg == 'ToM_self' or config.agent_alg == 'ToM_SB':
                torch_agent_actions = maddpg.step(torch_obs, torch_agent_actions, explore=True)
            else:
                torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # t2 = time.time()
            # print('time_step:', t2-t1)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions] #
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            obs_ep.append(obs)                  #episode_id,process, n_agents, dim
            agent_actions_ep.append(actions)    #episode_id, n_agents, process, dim
            rewards_ep.append(rewards)          #episode_id,process, n_agents,
            next_obs_ep.append(next_obs)            #episode_id,process, n_agents, dim
            dones_ep.append(dones)              #episode_id,process, n_agents,
            if config.agent_alg == 'ToM_SA' or config.agent_alg == 'ToM_S'  or config.agent_alg == 'ToM_self'or config.agent_alg == 'ToM_SB':
                replay_buffer.push(torch_agent_actions_pre, obs, agent_actions, rewards, next_obs, dones)
            else:
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads

            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                if config.n_episodes >300:
                    rollout = 2
                else:
                    rollout = config.n_rollout_threads
                for u_i in range(rollout):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        # t1 = time.time()
                        maddpg.update(sample, a_i, logger=logger)
                        # t2 = time.time()
                        # print('trian_time:', t2-t1, u_i, a_i)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew,
                              ep_i)
        logger.add_scalar('agent_mean/mean_episode_rewards',
                          np.mean(ep_rews),
                          ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')
        pbar.update(config.n_rollout_threads)

    pbar.close()
    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    for a_i, reward in enumerate(total_reward):
        reward_dir = str(log_dir) + '/agent{}/mean_episode_rewards'.format(a_i) + '/episode_rewards_{}'.format(config.cuda_num)
        os.makedirs(reward_dir)
        np.save(reward_dir, reward)


if __name__ == '__main__':
    config = get_common_args()
    # config.env_id = 'simple_tag'
    # # config.model_name = 'ma2c'
    config.agent_alg = 'ToM_SB'#
    config.adversary_alg = 'ToM_SB'
    #
    run(config)
