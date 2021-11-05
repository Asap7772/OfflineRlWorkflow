import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining,load_data_from_npy_chaining_mult
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, VQVAEEncoderConcatCNN, \
    ConcatBottleneckVQVAECNN, VQVAEEncoderCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger

import argparse, os
import roboverse
import numpy as np

import os
from os.path import expanduser

DEFAULT_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')

DEFAULT_PRIOR_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
DEFAULT_TASK_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
CUSTOM_LOG_DIR = '/home/asap7772/workflow_output'


def process_buffer(args):
    path = '/home/asap7772/cog_data/'
    buffers = []    
    home = os.path.expanduser("~")
    p_data_path =  os.path.join(home, 'prior_data/') if args.azure else '/nfs/kun1/users/asap7772/prior_data/' 
    ba = lambda x, p=args.prob, y=None: buffers.append((os.path.join(path, x),dict(p=p,alter_type=y,)))

    if args.buffer == 0:
        path = p_data_path
        ba('closed_drawer_prior.npy',y='zero')
        ba('drawer_task.npy')
    elif args.buffer == 1:
        path = p_data_path
        ba('blocked_drawer_1_prior.npy',y='zero')
        ba('drawer_task.npy')
    elif args.buffer == 2:
        path = p_data_path
        ba('blocked_drawer_2_prior.npy',y='zero')
        ba('drawer_task.npy')
    else:
        assert False, "Invalid Buffer"

    return buffers

def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    if variant['num_sample'] != 0:
        eval_env.num_obj_sample=variant['num_sample']
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size
    print(action_dim)

    cnn_params = variant['cnn_params']
    
    cnn_params.update(
        dropout = variant['dropout'],
        dropout_prob = variant['dropout_prob'],
    )

    if variant['bigger_net']:
        print('bigger_net')
        cnn_params.update(
            hidden_sizes=[1024, 512, 512, 512, 256],
        )
    if variant['deeper_net']:
        print('deeper conv net')
        cnn_params.update(
            kernel_sizes=[3, 3, 3, 3, 3],
            n_channels=[32, 32, 32, 32, 32],
            strides=[1, 1, 1, 1, 1],
            paddings=[1, 1, 1, 1, 1],
            pool_sizes=[2, 2, 1, 1, 1],
            pool_strides=[2, 2, 1, 1, 1],
            pool_paddings=[0, 0, 0, 0, 0]
        )
    
    if variant['smaller_net']:
        print('smaller conv net')
        cnn_params.update(
            kernel_sizes=[3],
            n_channels=[32],
            strides=[1],
            paddings=[1],
            pool_sizes=[2],
            pool_strides=[2],
            pool_paddings=[0],
            hidden_sizes=[16,],
        )

    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
        normalize_conv_activation=variant['normalize_conv_activation']
    )

    if variant['resnet_enc']:
            qf1 = VQVAEEncoderConcatCNN(**cnn_params) 
            qf2 = VQVAEEncoderConcatCNN(**cnn_params)
            target_qf1 = VQVAEEncoderConcatCNN(**cnn_params)
            target_qf2 = VQVAEEncoderConcatCNN(**cnn_params)
    else:
        if variant['bottleneck']:
            qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
            qf2 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
            if variant['share_encoder']:
                raise NotImplementedError
            target_qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
            target_qf2 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
        else:
            qf1 = ConcatCNN(**cnn_params)
            qf2 = ConcatCNN(**cnn_params)
            target_qf1 = ConcatCNN(**cnn_params)
            target_qf2 = ConcatCNN(**cnn_params)

    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())

    if variant['resnet_policy']:
        cnn_params.update(
            output_size=256,
            added_fc_input_size=0,
            hidden_sizes=[1024, 512],
            spectral_norm_fc=False,
            spectral_norm_conv=False,
            normalize_conv_activation=False,
        )

        policy_obs_processor = VQVAEEncoderCNN(**cnn_params)
    else:
        cnn_params.update(
            output_size=256,
            added_fc_input_size=0,
            hidden_sizes=[1024, 512],
            spectral_norm_fc=False,
            spectral_norm_conv=False,
            normalize_conv_activation=False,
        )
        policy_obs_processor = CNN(**cnn_params)

    policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
        shared_encoder=variant['share_encoder'],
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )

    observation_key = 'image'
    replay_buffer = load_data_from_npy_chaining(variant, expl_env, observation_key, duplicate=variant['duplicate'], num_traj=variant['num_traj'])

    # Translate 0/1 rewards to +4/+10 rewards.
    if variant['use_positive_rew']:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards * 6.0
            replay_buffer._rewards = replay_buffer._rewards + 4.0
        assert set(np.unique(replay_buffer._rewards)).issubset(
            set(6.0 * np.array([0, 1]) + 4.0))

    trainer = CQLTrainer(
        # enviroment
        env=eval_env,
        # networks
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        # l1/l2 regularization
        regularization=variant['regularization'],
        regularization_type=variant['regularization_type'],
        regularization_const=variant['regularization_const'],
        # bottleneck
        bottleneck=variant['bottleneck'],
        bottleneck_const=variant['bottleneck_const'],
        bottleneck_lagrange=variant['bottleneck_lagrange'],
        only_bottleneck = variant['only_bottleneck'],
        # dr3
        dr3=variant['dr3'],
        dr3_feat=variant['dr3_feat'],
        dr3_weight=variant['dr3_weight'],
        # logging
        log_dir = variant['log_dir'],
        wand_b=not variant['debug'],
        variant_dict=variant,
        validation=variant['val'],
        validation_buffer=None,
        # diagnostic
        no_td=variant['no_td'],
        no_data_qval=variant['no_data_qval'],
        # others
        **variant['trainer_kwargs']
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=False,
        batch_rl=True,
        **variant['algorithm_kwargs']
    )
    video_func = VideoSaveFunction(variant)
    algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return


if __name__ == "__main__":
    variant = dict(
        algorithm="CQL",
        version="normal",
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=300,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=30,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=10000,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=5.0,

            # lagrange
            with_lagrange=False,  # Defaults to False
            lagrange_thresh=5.0,

            # extra params
            num_random=1,
            max_q_backup=False,
            deterministic_backup=False,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[1, 1, 1],
            hidden_sizes=[1024, 512, 256],
            paddings=[1, 1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 1],
            pool_paddings=[0, 0, 0],
            image_augmentation=True,
            image_augmentation_padding=4,
            spectral_norm_conv=False,
            spectral_norm_fc=False,
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )
    
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--eval_every_n", default=1, type=int)
    parser.add_argument("--max-path-length", type=int, required=True)
    parser.add_argument("--num-eval-per-epoch", type=int, default=5)
    parser.add_argument('--eval_num', default=0, type=int)
    # replay buffer
    parser.add_argument("--use-positive-rew", action="store_true", default=False)
    parser.add_argument("--prior-buffer", type=str, default=DEFAULT_PRIOR_BUFFER)
    parser.add_argument("--task-buffer", type=str, default=DEFAULT_TASK_BUFFER)
    parser.add_argument("--duplicate", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    # reduce dataset size either with proportion of data or absolute number of trajectories
    parser.add_argument("--prob", default=1, type=float)
    parser.add_argument('--num_traj', default=0, type=int)
    #cql hyperparams
    parser.add_argument("--min-q-version", default=3, type=int,
                        help=("min_q_version = 3 (CQL(H)), "
                              "version = 2 (CQL(rho))"))
    parser.add_argument("--min-q-weight", default=1.0, type=float,
                        help="Value of alpha in CQL")
    parser.add_argument("--use-lagrange", action="store_true", default=False)
    parser.add_argument("--lagrange-thresh", default=5.0, type=float,
                        help="Value of tau, used with --use-lagrange")
    parser.add_argument("--max-q-backup", action="store_true", default=False,
                        help="For max_{a'} backups, set this to true")
    parser.add_argument("--no-deterministic-backup", action="store_true",
                        default=False,
                        help="By default, deterministic backup is used")
    parser.add_argument("--policy-eval-start", default=10000,
                        type=int)
    # architecture design choices
    parser.add_argument("--bigger_net", action="store_true", default=False)
    parser.add_argument('--smaller_net', action='store_true')
    parser.add_argument("--deeper_net", action="store_true", default=False)
    parser.add_argument("--resnet_enc", action="store_true", default=False)
    parser.add_argument("--resnet_policy", action="store_true", default=False)
    parser.add_argument("--share_encoder", action="store_true", default=False)
    parser.add_argument('--only_one', action='store_true')
    # dr3
    parser.add_argument("--dr3", action="store_true", default=False)
    parser.add_argument("--dr3_feat", action="store_true", default=False)
    parser.add_argument("--dr3_weight", default=0.001, type=float)
    # l1/l2 regularization
    parser.add_argument('--regularization', action='store_true')
    parser.add_argument('--regularization_type', type=str, default='l1')
    parser.add_argument('--regularization_const', type=float, default=1)
    # dropout
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    # bottleneck
    parser.add_argument("--deterministic_bottleneck", action="store_true", default=False)
    parser.add_argument("--only_bottleneck", action="store_true", default=False)
    parser.add_argument("--bottleneck", action='store_true')
    parser.add_argument('--bottleneck_const', type=float, default=0.5)
    parser.add_argument('--bottleneck_dim', type=int, default=16)
    parser.add_argument('--bottleneck_lagrange', action='store_true')
    # diagnostics
    parser.add_argument('--normalize_conv_activation', action='store_true')
    parser.add_argument('--no_td', action='store_true')
    parser.add_argument('--no_data_qval', action='store_true')
    parser.add_argument('--clip_grad_val', type=float, default=10)
    #other hyperparams
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--policy-lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--name", default='test', type=str)

    args = parser.parse_args()
    enable_gpus(args.gpu)

    variant['no_td'] = args.no_td
    variant['no_data_qval'] = args.no_data_qval
    variant['clip_grad_val'] = args.clip_grad_val
    
    variant['smaller_net'] = args.smaller_net
    variant['regularization'] = args.regularization
    variant['regularization_type'] = args.regularization_type
    variant['regularization_const'] = args.regularization_const

    variant['dropout'] = args.dropout
    variant['dropout_prob'] = args.dropout_prob

    variant['trainer_kwargs']['discount'] = args.discount
    variant['bigger_net'] = args.bigger_net
    variant['deeper_net'] = args.deeper_net
    variant['resnet_enc'] = args.resnet_enc
    variant['resnet_policy'] = args.resnet_policy
    variant['share_encoder'] = args.share_encoder
    variant['normalize_conv_activation'] = args.normalize_conv_activation

    variant['spectral_norm_conv'] = args.spectral_norm_conv
    variant['spectral_norm_fc'] = args.spectral_norm_fc

    variant['env'] = args.env
    variant['val'] = args.val
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = \
        args.num_eval_per_epoch*args.max_path_length
    variant['algorithm_kwargs']['eval_every_n_epochs'] = args.eval_every_n


    variant['prior_buffer'] = args.prior_buffer
    variant['task_buffer'] = args.task_buffer
    
    variant['bottleneck'] = args.bottleneck
    variant['bottleneck_const'] = args.bottleneck_const

    variant['dr3'] = args.dr3
    variant['dr3_feat'] = args.dr3_feat
    variant['dr3_weight'] = args.dr3_weight

    variant['bottleneck_lagrange'] = args.bottleneck_lagrange
    variant['bottleneck_dim'] = args.bottleneck_dim
    variant['deterministic_bottleneck']=args.deterministic_bottleneck
    variant['only_bottleneck'] = args.only_bottleneck
    variant['num_traj'] = args.num_traj
    variant['num_sample'] = args.eval_num
    variant['trainer_kwargs']['discount'] = args.discount
    
    variant['debug'] = False

    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs']['deterministic_backup'] = \
        not args.no_deterministic_backup
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    variant['trainer_kwargs']['with_lagrange'] = args.use_lagrange
    variant['duplicate'] = args.duplicate

    # Translate 0/1 rewards to +4/+10 rewards.
    variant['use_positive_rew'] = args.use_positive_rew
    variant['seed'] = args.seed

    buffers = process_buffer(args)
    variant['buffer'] = buffers
    variant['bufferidx'] = args.buffer
    variant['prior_buffer'] = buffers[0]
    variant['task_buffer'] = buffers[1]

    ptu.set_gpu_mode(True)
    exp_prefix = 'cql-cog-{}'.format(args.env)
    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None
    
    variant['base_log_dir'] = base_log_dir
    
    log_dir = setup_logger(args.name, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    variant['log_dir'] = log_dir
    experiment(variant)
