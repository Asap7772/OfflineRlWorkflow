import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql_latent import CQLTrainerLatent
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger
from rlkit.data_management.load_buffer_real import *

import argparse, os
import roboverse
import numpy as np
from os.path import expanduser

DEFAULT_PRIOR_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
DEFAULT_TASK_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
CUSTOM_LOG_DIR = '/nfs/kun1/users/asap7772/cog/data/'


def experiment(variant):
    eval_env = DummyEnv()
    expl_env = eval_env
    action_dim = eval_env.action_space.low.size

    M = 512
    obs_dim = variant['obs_dim']
    mlp_params = dict(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M]*3,
    )
    qf1 = ConcatMlp(**mlp_params)
    qf2 = ConcatMlp(**mlp_params)
    target_qf1 = ConcatMlp(**mlp_params)
    target_qf2 = ConcatMlp(**mlp_params)

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M]*4,
        obs_processor=None,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    
    paths = []
    observation_key = 'image'
    data_path = os.path.join(expanduser("~"),'val_data_relabeled') if args.azure else '/nfs/kun1/users/asap7772/real_data_drawer/val_data_relabeled/'
    if args.buffer == 0:
        print('lid on')
        paths.append((os.path.join(data_path,'fixed_pot_demos_latent.npy'), os.path.join(data_path,'fixed_pot_demos_lidon_rew_handlabel_06_13.pkl')))
    elif args.buffer == 1:
        print('lid off')
        paths.append((os.path.join(data_path,'fixed_pot_demos_latent.npy'), os.path.join(data_path,'fixed_pot_demos_lidoff_rew_handlabel_06_13.pkl')))
    elif args.buffer == 2:
        print('tray')
        paths.append((os.path.join(data_path,'fixed_tray_demos_latent.npy'), os.path.join(data_path,'fixed_tray_demos_rew.pkl')))
    elif args.buffer == 3:
        print('drawer')
        paths.append((os.path.join(data_path,'fixed_drawer_demos_latent.npy'), os.path.join(data_path,'fixed_drawer_demos_draweropen_rew_handlabel_06_13.pkl')))
    elif args.buffer == 4:
        print('Stephen Tool Use')
        path = '/nfs/kun1/users/stephentian/on_policy_longer_1_26_buffers/move_tool_obj_together_fixed_6_2_train.pkl'
    else:
        assert False

    if args.buffer in [4]:
        print('loading')
        replay_buffer = pickle.load(open(path,'rb'))
        print('done loading')
        import ipdb; ipdb.set_trace()
        if variant['rew_type'] == 1:
            pass
        elif variant['rew_type'] == 2:
            pass
    else:
        replay_buffer = get_buffer(observation_key=observation_key, color_jitter = variant['color_jitter'])
        for path, rew_path in paths:
            load_path(path, rew_path, replay_buffer,bc=variant['filter'])

        # Translate 0/1 rewards to +0/+10 rewards.
        replay_buffer._rewards = replay_buffer._rewards * 10.0

    trainer = CQLTrainerLatent(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            log_dir = variant['log_dir'],
            variant_dict=variant,
            real_data=True,
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
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        algorithm_kwargs=dict(
            # num_epochs=100,
            # num_eval_steps_per_epoch=50,
            # num_trains_per_train_loop=100,
            # num_expl_steps_per_train_loop=100,
            # min_num_steps_before_training=100,
            # max_path_length=10,
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
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--filter', action='store_true', default=False)
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
    parser.add_argument("--policy-lr", default=1e-4, type=float)
    parser.add_argument("--min-q-version", default=3, type=int,
                        help=("min_q_version = 3 (CQL(H)), "
                              "version = 2 (CQL(rho))"))
    parser.add_argument("--num-eval-per-epoch", type=int, default=5)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument('--buffer', type=str, default='0')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--obs_dim', type=int, default=720)
    parser.add_argument('--color_jitter', action='store_true')
    parser.add_argument('--azure', action='store_true')
    parser.add_argument('--rew_type', type=int, default=0)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    
    variant['buffer'] = args.buffer = int(args.buffer)
    variant['rew_type'] = args.rew_type
    variant['filter'] = args.filter
    variant['color_jitter'] = args.color_jitter
    variant['obs_dim'] = args.obs_dim
    variant['algorithm_kwargs']['max_path_length'] = 0
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 0

    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs']['deterministic_backup'] = \
        not args.no_deterministic_backup
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    variant['trainer_kwargs']['with_lagrange'] = args.use_lagrange

    variant['seed'] = args.seed

    ptu.set_gpu_mode(True)
    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None
    
    variant['base_log_dir'] = base_log_dir
    
    log_dir = setup_logger(args.name, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    variant['log_dir'] = log_dir
    experiment(variant)