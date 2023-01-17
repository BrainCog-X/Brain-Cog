import argparse

def get_common_args():
    parser = argparse.ArgumentParser()

    ## multiprocessing
    parser.add_argument('--process', type=int, default=5, help='multiprocessing')

    ## the environment setting  'CLASSIC', 'HUNT', 'HARVEST', 'ESCALATION'
    parser.add_argument('--ENV', type=str, default='HUNT', help='the version of the game, choose from ["CLASSIC", "HUNT", "HARVEST", "ESCALATION"]')
    parser.add_argument('--env_name', type=str, default='stag_stay', help='the version of the game, choose from ["CLASSIC", "HUNT", "HARVEST", "ESCALATION"]')

    parser.add_argument('--obs_type', type=str, default='coords', help='Can be "image" for pixel-array based observations, or "coords" for just the entity coordinates')
    parser.add_argument('--forage_quantity', type=int, default=2, help='the number of trees')
    parser.add_argument('--opponent_policy', type=str, default='random', help='the poliocy of opponent')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')

    ## The alternative policy   ################################################
    parser.add_argument('--num_run', type=int, default='4', help='the number of run')
    ## 'svdn',  'stomvdn'
    parser.add_argument('--alg', type=str, default='svdn', help='the algorithm to train the agent')
    parser.add_argument('--mode', type=str, default='train', help='the mode')
    parser.add_argument('--n_steps', type=int, default=1000000, help='total time steps')#2000000
    parser.add_argument('--n_episodes', type=int, default=2, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network_sc for all agents_sc')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon  factor')
    ############# "Adam"
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=10, help='how often to evaluate the model')#5000
    parser.add_argument('--evaluate_epoch', type=int, default=6, help='number of the epoch to evaluate the agent')#32

    ## save weights->model/args->log/reward->result/plot
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy_base')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy_base')#./result#/home/zhaozhuoya/exp2/ToM2_test/result
    parser.add_argument('--log_dir', type=str, default='./log', help='args directory')
    parser.add_argument('--plot_dir', type=str, default='./plot', help='args directory')

    parser.add_argument('--exp_dir', type=str, default='/exp_vdn', help='result directory of the policy_base')
    parser.add_argument('--save_model_dir', type=str, default='/199_rnn_net_params_hunt1.pkl', help='load weights and bias')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')   #True
    parser.add_argument('--mini_batch_size', type=int, default=250, help='whether to use the GPU')
    args = parser.parse_args()
    parser.add_argument('--device', type=str, default='cuda:{}'.format(args.num_run), help='whether to use the GPU')  #'cuda:1'
    args = parser.parse_args()
    return args

# arguments of coma
def get_coma_args(args):
    # network_sc
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    # args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args

# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network_sc
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.ppo_hidden_size = 64
    args.lr = 5e-4

    # epsilon greedy
    # args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # MAVEN
    args.noise_dim = 16
    args.lambda_mi = 0.001
    args.lambda_ql = 1
    args.entropy_coefficient = 0.001
    return args
