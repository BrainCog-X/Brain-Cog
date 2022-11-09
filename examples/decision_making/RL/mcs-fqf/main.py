import argparse
import os
import pprint
import numpy as np
import torch
from network import SpikingDQN
from ..atari.atari_wrapper import wrap_deepmind
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, SequenceLogger
from discrete import SpikeFractionProposalNetwork, SpikeFullQuantileFunction
from policy import FQFPolicy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='MsPacmanNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=3128)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fraction-lr', type=float, default=2.5e-9)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num-fractions', type=int, default=32)
    parser.add_argument('--num-cosines', type=int, default=64)
    parser.add_argument('--ent-coef', type=float, default=10.)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512])
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=100000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    # parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--frames-stack', type=int, default=4)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--resume-id', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    parser.add_argument('--save-buffer-name', type=str, default=None)

    parser.add_argument('--time-window', type=int, default=8)
    parser.add_argument('--prefix', type=str, default='')

    parser.add_argument('--save-interval', type=int, default=10)

    
    return parser.parse_args()


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)


def make_atari_env_watch(args):
    return wrap_deepmind(
        args.task,
        frame_stack=args.frames_stack,
        episode_life=False,
        clip_rewards=False
    )


def main(args=get_args()):
    print('Setting: ', args)
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print('update_per_step: ', args.update_per_step)
    print('lr: ', args.lr)
    # make environments
    train_envs = ShmemVectorEnv(
        [lambda: make_atari_env(args) for _ in range(args.training_num)]
    )
    test_envs = ShmemVectorEnv(
        [lambda: make_atari_env_watch(args) for _ in range(args.test_num)]
    )
    # define model
    feature_net = SpikingDQN(
        *args.state_shape, args.action_shape, args.device, time_window=args.time_window, features_only=True
    )
    net = SpikeFullQuantileFunction(
        feature_net,
        args.action_shape,
        args.hidden_sizes,
        args.num_cosines,
        device=args.device,
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    fraction_net = SpikeFractionProposalNetwork(args.num_fractions, net.input_dim)
    fraction_optim = torch.optim.RMSprop(
        fraction_net.parameters(), lr=args.fraction_lr
    )
    # define policy
    policy = FQFPolicy(
        net,
        optim,
        fraction_net,
        fraction_optim,
        args.gamma,
        args.num_fractions,
        args.ent_coef,
        args.n_step,
        target_update_freq=args.target_update_freq
    ).to(args.device)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack
    )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # log
    log_path = os.path.join(args.logdir, args.task, 'spike_fqf', args.prefix)
    model_log_path = os.path.join(log_path, 'models')
    if not os.path.exists(model_log_path):
        os.makedirs(model_log_path)
    print('log_path: ', log_path)
   
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, save_interval=args.save_interval)
    result_logger = SequenceLogger(log_path)

    def save_checkpoint_fn(epoch, env_step, gradient_step, epoch_round=True):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        if epoch_round:
            ckpt_path = os.path.join(model_log_path, 'checkpoint_epoch{}.pth'.format(epoch))
        else:
            ckpt_path = os.path.join(model_log_path, 'checkpoint.pth')
        ckpt = {
            'epoch': epoch,
            'env_step': env_step,
            'gradient_step': gradient_step,
            'model': policy.state_dict()
        }
        torch.save(ckpt, ckpt_path)
        return ckpt_path

    setting_path = os.path.join(log_path, 'settings.txt')
    argsDict = args.__dict__
    with open(setting_path, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
        

    def save_fn(policy, is_best=False):
        if is_best:
            torch.save(policy.state_dict(), os.path.join(log_path, 'best_policy.pth'))
        else:
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif 'Pong' in args.task:
            return mean_rewards >= 20
        else:
            return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    if args.watch:
        watch()
        exit(0)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn,
        result_logger=result_logger
    )

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    main(get_args())
