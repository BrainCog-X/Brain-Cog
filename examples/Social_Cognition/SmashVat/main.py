import argparse
import os
import random
import time

import numpy as np
import torch

from environment import *
from dqn import AnseEmpDQN

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=1919810)
parser.add_argument('--env', type=str, default='BasicHumanVatGoalEnv')
parser.add_argument('--net-type', type=str, default='ANN')

parser.add_argument('--init-buffer-size', type=int, default=10000)
parser.add_argument('--replay-buffer-size', type=int, default=100000)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--target-update-interval', type=int, default=1000)

parser.add_argument('--weight-sep', type=float, default=20)
parser.add_argument('--weight-emp-impact', type=float, default=None)

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num-episodes', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--epsilon-start', type=float, default=1.0)
parser.add_argument('--epsilon-end', type=float, default=0.01)
parser.add_argument('--decay-start', type=float, default=0.05)
parser.add_argument('--decay-end', type=float, default=0.95)
parser.add_argument('--checkpoint-interval', type=int, default=1000)

parser.add_argument('--log-dir', type=str, default=None)
parser.add_argument('--model-save-dir', type=str, default=None)
parser.add_argument('--gif-dir', type=str, default=None)


def set_seed(seed=114514):
    random.seed(seed)  # replay buffer 中使用了random.sample
    np.random.seed(seed)  # e-greedy 中使用了np.random_choice
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_args(args, log_dir):
    filename = os.path.join(log_dir, 'args.txt')
    with open(filename, 'w') as file:
        for arg in vars(args):
            file.write('{}: {}\n'.format(arg, getattr(args, arg)))


def make_dirs(args, timestamp):
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join('./logs', args.net_type + '-' + args.env, timestamp)
        os.makedirs(log_dir, exist_ok=True)

    if args.model_save_dir is not None:
        model_save_dir = args.model_save_dir
    else:
        model_save_dir = os.path.join('./models', args.net_type + '-' + args.env, timestamp)
        os.makedirs(model_save_dir, exist_ok=True)

    if args.gif_dir is not None:
        gif_dir = args.gif_dir
    else:
        gif_dir = os.path.join('./gifs', args.net_type + '-' + args.env)
        os.makedirs(gif_dir, exist_ok=True)

    return log_dir, model_save_dir, gif_dir


def main():
    args = parser.parse_args()
    set_seed(args.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir, model_save_dir, gif_dir = make_dirs(args, timestamp)
    save_args(args, log_dir)

    assert args.env in env_list
    env = eval(args.env)()
    with torch.cuda.device(args.cuda):
        model = AnseEmpDQN(env,
                           net_type=args.net_type,
                           init_buffer_size=args.init_buffer_size,
                           replay_buffer_size=args.replay_buffer_size,
                           batch_size=args.batch_size,
                           target_update_interval=args.target_update_interval,
                           weight_sep=args.weight_sep,
                           weight_emp_impact=args.weight_emp_impact)
        model.train(lr=args.lr, num_episodes=args.num_episodes, gamma=args.gamma,
                    epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                    decay_start=args.decay_start, decay_end=args.decay_end,
                    checkpoint_interval=args.checkpoint_interval, checkpoint_dir=model_save_dir,
                    log_dir=log_dir)
        model.save(model_save_dir)

        gif_name = os.path.join(gif_dir, '{}.gif'.format(timestamp))
        model.run(gif_name=gif_name)
        model.env.close()


if __name__ == '__main__':
    main()
