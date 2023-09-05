import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.tom11 import ToM_decision11
from algorithms.maddpg import MADDPG
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv

def display_frames_as_gif(frames):
    patch = plt.imshow(frames[1])
    plt.axis('off')

    plt.savefig('./images/comm2', bbox_inches='tight')

def make_parallel_env(env_id, n_rollout_threads, discrete_action, num_good_agents, num_adversaries):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, num_good_agents=num_good_agents, num_adversaries=num_adversaries, discrete_action=discrete_action)
            # env.seed(seed + rank * 1000)
            # np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    rew_ep = []
    for i in range(config.num):
        for run_num in config.run_num:
            pbar = tqdm(config.n_episodes)
            model_path = (Path('./models') / config.env_id / config.model_name /
                          ('run%i' % (run_num)))
            if config.incremental is not None:
                model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                           config.incremental)
            else:
                model_path = model_path / 'model.pt'

            # if config.save_gifs:
            #     gif_path = model_path.parent / 'gifs'
            #     gif_path.mkdir(exist_ok=True)
            if config.alg == 'ToM1':
                maddpg = ToM_decision11.init_from_save(model_path)
            elif config.alg == 'ToM_SB01' or config.alg== 'ToM_SA01':
                maddpg = ToM_decision01.init_from_save(model_path)#.eval()
            elif config.alg == 'ToM_SBN1' or config.alg== 'ToM_SAN1':
                maddpg = ToM_decisionN1.init_from_save(model_path)#.eval()
            elif config.alg == 'MADDPG':
                maddpg = MADDPG.init_from_save(model_path)

            # env = make_env(config.env_id, num_good_agents=config.num_good_agents,
            #                num_adversaries=config.num_adversaries, discrete_action=maddpg.discrete_action)
            env = make_parallel_env(config.env_id, config.n_rollout_threads,
                                    config.discrete_action, config.num_good_agents, config.num_adversaries)
            maddpg.prep_rollouts(device='cpu')
            ifi = 1 / config.fps  # inter-frame interval

            for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
                rew = np.zeros((config.n_rollout_threads, config.num_good_agents + config.num_adversaries))
                torch_agent_actions = [torch.zeros((config.n_rollout_threads, 5))
                                       for i in range(maddpg.nagents)]
                # print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
                obs = env.reset()

                for t_i in range(config.episode_length):
                    # calc_start = time.time()
                    # rearrange observations to be per agent, and convert to torch Variable
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                          requires_grad=False)
                                 for i in range(maddpg.nagents)]
                    # get actions as torch Variables
                    # t1 = time.time()
                    if config.alg == 'MADDPG':
                        torch_actions = maddpg.step(torch_obs, explore=False)

                    else:
                        torch_actions = maddpg.step(torch_obs, torch_agent_actions, explore=False)

                    actions = [ac.data.cpu().numpy() for ac in torch_actions]
                    actions = [[ac[i] for ac in actions] for i in range(config.n_rollout_threads)]
                    obs, rewards, dones, infos = env.step(actions)
                    rew += rewards
                rew_ep.append(rew)
                pbar.update(config.n_rollout_threads)

            pbar.close()
    rew_ep = np.concatenate(rew_ep, 0)
    rew_ep_agent = rew_ep.mean(0)
    std_ep_agent = rew_ep.std(0)
    print('mean:', rew_ep_agent, 'std:', std_ep_agent)
    rew_ep_good = rew_ep[:, -config.num_good_agents:].sum(1).mean()
    rew_ep_adv = rew_ep[:, :config.num_adversaries].sum(1).mean()
    std_ep_good = rew_ep[:, -config.num_good_agents:].sum(1).std()
    std_ep_adv = rew_ep[:, :config.num_adversaries].sum(1).std()
    print('good:', rew_ep_good, 'std:', std_ep_good)
    print('adv:', rew_ep_adv, 'std:', std_ep_adv)
    rew_ep_all = rew_ep.sum(1).mean()
    std_ep_all = rew_ep.sum(1).std()
    print('all:', rew_ep_all, 'std:', std_ep_all)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg",
                        default="ToM_SAN1", type=str,
                        choices=['MADDPG', 'ToM_SB01', 'ToM_SA01', 'ToM_SBN1', 'ToM_SAN1',
                            'ToM1'])
    parser.add_argument("--env_id", default='simple_adversary', type=str, help="Name of environment",
                        choices=['simple_tag', 'simple_world_comm', 'hetero_spread',
                                 'simple_adversary', 'simple_spread'
                                 ])
    parser.add_argument("--num_good_agents", default=None, type=int,
                        help="Num of Agent")
    parser.add_argument("--num_adversaries", default=None, type=int,
                        help="Num of Adversary")
    parser.add_argument("--model_name", default='4VS2_tomaAN1', type=str,
                        help="Name of model")   #ma2c, maddpg, maddpg_rnn
    parser.add_argument("--run_num", default=2, type=int, nargs='+')
    parser.add_argument("--save_gifs", default=True, action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default= None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=1000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--eval",
                        default=True, type=bool,
                        )
    parser.add_argument("--num", default=3, type=int )
    parser.add_argument("--n_rollout_threads", default=20, type=int)
    parser.add_argument("--discrete_action",
                        # default=False, type=bool,
                        action='store_true')


    config = parser.parse_args()


    if 'ToM_SB' in config.alg:
        config.agent_alg = 'without_tom'
        config.adversary_alg = 'with_tom'
    elif 'ToM_SA' in config.alg:
        config.agent_alg = 'with_tom'
        config.adversary_alg = 'without_tom'
    elif config.alg == 'ToM1':
        config.agent_alg = 'with_tom'
        config.adversary_alg = 'with_tom'
    else:
        config.agent_alg = config.alg
        config.adversary_alg = config.alg

    if config.num_good_agents == None and config.num_adversaries == None:
        if config.env_id == 'simple_adversary':
            config.num_good_agents = 2
            config.num_adversaries = 1
        elif config.env_id == 'simple_tag':
            config.num_good_agents = 2
            config.num_adversaries = 2
        elif config.env_id == 'simple_world_comm':
            config.num_good_agents = 2
            config.num_adversaries = 4
        elif config.env_id == 'simple_spread':
            config.num_good_agents = 3
            config.num_adversaries = 0
    run(config)
