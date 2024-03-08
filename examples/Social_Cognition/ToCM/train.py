import argparse
import os
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from agent.runners.ToCMRunner import ToCMRunner
from configs import Experiment, SimpleObservationConfig, NearRewardConfig, DeadlockPunishmentConfig, \
    RewardsComposerConfig
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig, MPEConfig  # TODO
from configs.ToCM.ToCMControllerConfig import ToCMControllerConfig
from configs.ToCM.ToCMLearnerConfig import ToCMLearnerConfig
from environments import Env
from utils.util import get_dim_from_space, get_cent_act_dim
import torch
import random
import numpy as np
import setproctitle
setproctitle.setproctitle("MPE_obs_2_hetero")
def occumpy_mem(cuda_device):
    total, used = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")[int(cuda_device)].split(',')
    total = int(total)
    used = int(used)
    cc = 0.85
    block_mem = int((total - used) * cc)
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="mpe", help='starcraft or mpe')  # TODO
    parser.add_argument('--env_name', type=str, default="hetero_spread", help='Specific setting')  # TODO
    # star : 2s_vs_1sc MMM 2s3z 3s_vs_3z 3s5z_vs_3s6z simple_spread
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=50, help='random')
    # TODO num_landmarks num_adversaries episode_length num_good_agents
    parser.add_argument('--num_agents', type=int, default=2, help='mpe_num_agents')  # simple_adversary
    parser.add_argument('--num_adversaries', type=int, default=None, help='mpe_num_adversaries')
    parser.add_argument('--num_good_agents', type=int, default=None, help='mpe_num_good_agents')
    parser.add_argument('--num_landmarks', type=int, default=2, help='mpe_num_landmarks')
    parser.add_argument('--episode_length', type=int, default=25, help='mpe_episode_length')
    parser.add_argument('--num_rollout_threads', type=int, default=128, help='mpe_episode_length')
    parser.add_argument('--benchmark', type=bool, default=False, help='mpe_use_benchmark')
    return parser.parse_args()  # 为啥直接跳到prepare_starcraft_configs函数里了


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_ToCM(exp, n_workers):  # no env.episode_length
    runner = ToCMRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.run(exp.steps, exp.episodes)  # 10 ** 10 50000


def get_env_info(configs, env):
    for config in configs:
        config.IN_DIM = env.n_obs  # 17 2s_vs_1sc
        config.ACTION_SIZE = env.n_actions  # 7 2s_vs_1sc
    env.close()


def get_env_info_mpe(configs, env):  # add to ToCM controller and worker
    # TODO cent_obs_dim and cent_action_dim  use share_policy
    for config in configs:
        config.CENT_OBS_DIM = get_dim_from_space(env.env.share_observation_space[0])  # 54=num_agents*IN_DIM
        config.CENT_ACT_DIM = get_cent_act_dim(env.env.action_space)  # 15=num_agents*ACTION_SIZE
        config.IN_DIM = get_dim_from_space(env.env.observation_space[0])  # dim 18
        config.ACTION_SIZE = get_dim_from_space(env.env.action_space[0])  # dim 5
    env.close()




def prepare_starcraft_configs(env_name, device):
    # env_name '3s5z_vs_3s6z'  device 'cuda:6'  RANDOM_SEED 1   args.n_workers 2
    # args.env 'starcraft'    args.env_name '3s5z_vs_3s6z'    args.device 'cuda:6'    args.seed 1
    agent_configs = [ToCMControllerConfig(env_name, RANDOM_SEED, device),
                     ToCMLearnerConfig(env_name, RANDOM_SEED, device)]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}


def prepare_mpe_configs(arg):
    agent_configs = [ToCMControllerConfig(arg.env_name, RANDOM_SEED, arg.device),
                     ToCMLearnerConfig(arg.env_name, RANDOM_SEED, arg.device)]
    env_config = MPEConfig(arg)
    get_env_info_mpe(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}  # TODO whether has reward config and obs builder config



if __name__ == "__main__":
    # occumpy_mem(2)
    # RANDOM_SEED = 23  # RANDOM_SEED 1
    args = parse_args()
    # print("args=", args)
    RANDOM_SEED = args.seed
    setup_seed(RANDOM_SEED)  # TODO
    # args.env_name '3s5z_vs_3s6z' args.device 'cuda:6'  args.seed 1  args.n_workers 2
    if args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name, args.device)
    elif args.env == Env.MPE:
        configs = prepare_mpe_configs(args)
        # as env is mpe env_name is simple_adversary
    else:
        raise Exception("Unknown environment")

    configs["env_config"][0].ENV_TYPE = Env(args.env)  # 转化为字符串
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    exp = Experiment(steps=10 ** 10,
                     episodes=50000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env), args.device,  # TODO
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])
    # print("exp=", exp)
    train_ToCM(exp, n_workers=args.n_workers)
