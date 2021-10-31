import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy, load_data_from_npy_mult, load_data_from_npy_chaining
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector, CustomMDPPathCollector_EVAL

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.bc import BCTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN, VQVAEEncoderCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger
from rlkit.envs.dummy_env import DummyEnv
import pickle
from rlkit.data_management.load_buffer_real import *

import argparse, os
import roboverse

# '/media/avi/data/Work/github/avisingh599/minibullet/data/'
#                   'oct6_Widow250DrawerGraspNeutral-v0_20K_save_all_noise_0.1'
#                   '_2020-10-06T19-37-26_100.npy'

# DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')

DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')
CUSTOM_LOG_DIR = '/home/stian/doodad-output'


def experiment(variant):
    variant['image_shape'] =  (48,48,3) if variant['small_image'] else (64,64,3)
    eval_env = expl_env = DummyEnv(image_shape=variant['image_shape'])
    action_dim = eval_env.action_space.low.size

    if variant['multi_bin']:
        eval_env.multi_tray = True
        expl_env.multi_tray = False

    cnn_params = variant['cnn_params']
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
    cnn_params.update(
        input_width=48 if variant['small_image'] else 64,
        input_height=48 if variant['small_image'] else 64,
        input_channels=3,
        output_size=1,
        added_fc_input_size= action_dim,
    )
    # qf1 = ConcatCNN(**cnn_params)
    # qf2 = ConcatCNN(**cnn_params)
    # target_qf1 = ConcatCNN(**cnn_params)
    # target_qf2 = ConcatCNN(**cnn_params)
    
    cnn_params.update(
        output_size=256,
        added_fc_input_size=variant['state_dim'] if variant['imgstate'] else 0,
        hidden_sizes=[1024, 512],
    )

    if variant['vqvae_enc']:
        policy_obs_processor = VQVAEEncoderCNN(**cnn_params)
    else:
        policy_obs_processor = CNN(**cnn_params)

    policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )
    
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector_EVAL(
        eval_env,
        eval_policy,
    )

    observation_key = ('image', 'state') if variant['imgstate'] else 'image'
    paths = []
    if args.azure:
        data_path = '/home/asap7772/drawer_data'
    else:
        data_path = '/nfs/kun1/users/ashvin/data/val_data'
    if args.buffer == 0:
        print('lid on')
        paths.append((os.path.join(data_path,'fixed_pot_demos.npy'), os.path.join(data_path,'fixed_pot_demos_putlidon_rew.pkl')))
    elif args.buffer == 1:
        print('lid off')
        paths.append((os.path.join(data_path,'fixed_pot_demos.npy'), os.path.join(data_path,'fixed_pot_demos_takeofflid_rew.pkl')))
    elif args.buffer == 2:
        print('tray')
        paths.append((os.path.join(data_path,'fixed_tray_demos.npy'), os.path.join(data_path,'fixed_tray_demos_rew.pkl')))
    elif args.buffer == 3:
        print('drawer')
        paths.append((os.path.join(data_path,'fixed_drawer_demos.npy'), os.path.join(data_path,'fixed_drawer_demos_rew.pkl')))
    elif args.buffer == 4:
        print('Stephen Tool Use')
        path = '/nfs/kun1/users/stephentian/on_policy_longer_1_26_buffers/move_tool_obj_together_fixed_6_2_train.pkl'
    elif args.buffer == 5:
        print('Albert Pick Place')
        px = os.path.join(os.path.expanduser("~"),'val_data_relabeled', 'combined_2021-06-03_21_36_48_labeled.pkl') if args.azure else '/nfs/kun1/users/albert/realrobot_datasets/combined_2021-06-03_21_36_48_labeled.pkl'
        data_path = '/nfs/kun1/users/albert/realrobot_datasets/combined_2021-06-03_21_36_48_labeled.pkl'
        if args.azure:
            data_path = px
        paths.append((data_path, None))
    else:
        assert False
    
    if args.buffer in [4]:
        replay_buffer = pickle.load(open(path,'rb'))
    else:
        replay_buffer = get_buffer(observation_key=observation_key, image_shape=variant['image_shape'])
        for path, rew_path in paths:
            load_path(path, rew_path, replay_buffer, small_img=variant['small_image'], bc=True, imgstate = variant['imgstate'])
    
    trainer = BCTrainer(
        env=eval_env,
        policy=policy,
        #qf1=qf1,
        #qf2=qf2,
        #target_qf1=target_qf1,
        #target_qf2=target_qf2,
        dist_diff=variant['dist_diff'],
        log_dir=variant['log_dir'],
        imgstate=variant['imgstate'],
        variant_dict=variant,
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
        algorithm="BC",
        version="normal",
        algorithm_kwargs=dict(
            # num_epochs=100,
            # num_eval_steps_per_epoch=50,
            # num_trains_per_train_loop=100,
            # num_expl_steps_per_train_loop=100,
            # min_num_steps_before_training=100,
            # max_path_length=10,
            num_epochs=3000,
            num_eval_steps_per_epoch=5,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=5,
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
            lagrange_thresh=10.0,

            # extra params
            num_random=1,
            max_q_backup=False,
            deterministic_backup=False,
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-path-length", type=int, default=1)
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--min-q-weight", default=1.0, type=float,
                        help="Value of alpha in CQL")
    parser.add_argument("--use-lagrange", action="store_true", default=False)
    parser.add_argument("--lagrange-thresh", default=5.0, type=float,
                        help="Value of tau, used with --use-lagrange")
    parser.add_argument("--use-positive-rew", action="store_true",
                        default=False)
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
    parser.add_argument("--name", default='test', type=str)
    parser.add_argument("--bin_color", action="store_true", default=False)
    parser.add_argument("--multi_bin", action="store_true", default=False)
    parser.add_argument("--mixture", action="store_true", default=False)
    parser.add_argument("--transfer", action="store_true", default=False)
    parser.add_argument("--transfer_multiview", action="store_true", default=False)
    parser.add_argument("--chaining", action="store_true", default=False)
    parser.add_argument("--p", default=0.2, type=float)
    parser.add_argument("--prob", default=1.0, type=float)
    parser.add_argument('--segment_type', default='fixed_other', type = str)
    parser.add_argument('--eval_multiview', default='single', type = str)
    parser.add_argument('--dist_diff', action="store_true", default=False)

    parser.add_argument('--larger_net', action="store_true", default=False)
    
    # Stephen added
    parser.add_argument('--deeper_net', action="store_true", default=False)
    parser.add_argument('--azure', action="store_true", default=False)
    parser.add_argument('--vqvae_enc', action="store_true", default=False)
    parser.add_argument('--duplicate', action="store_true", default=False)
    parser.add_argument('--num_traj', default=0, type=int)
    parser.add_argument('--smimg', default=False, action='store_true')
    parser.add_argument('--imgstate', default=False, action='store_true') # both image and state
    parser.add_argument('--state_dim', default=3, type=int)

    args = parser.parse_args()
    variant['state_dim'] = args.state_dim
    variant['imgstate'] = args.imgstate
    variant['transfer'] = args.transfer
    variant['mixture'] = args.mixture
    variant['chaining'] = args.chaining
    variant['p'] = args.p
    variant['bin'] = args.bin_color
    variant['segment_type'] = args.segment_type
    variant['small_image'] = args.smimg

    variant['transfer_multiview'] = args.transfer_multiview
    variant['eval_multiview'] = args.eval_multiview
    variant['dist_diff'] = args.dist_diff

    variant['deeper_net'] = args.deeper_net
    variant['vqvae_enc'] = args.vqvae_enc
    variant['duplicate'] = args.duplicate
    variant['num_traj'] = args.num_traj

    if args.buffer.isnumeric():
        args.buffer = int(args.buffer)

    enable_gpus(args.gpu)
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = \
        args.num_eval_per_epoch*args.max_path_length

    variant['buffer'] = args.buffer

    variant['trainer_kwargs']['max_q_backup'] = args.max_q_backup
    variant['trainer_kwargs']['deterministic_backup'] = \
        not args.no_deterministic_backup
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    variant['trainer_kwargs']['with_lagrange'] = args.use_lagrange
    variant['multi_bin'] = args.multi_bin

    if args.larger_net:
        variant['cnn_params'] = dict(
            kernel_sizes=[3, 3, 3, 3, 3, 3],
            n_channels=[16, 16, 16, 16,16,16],
            strides=[1, 1, 1, 1, 1, 1],
            hidden_sizes=[1024, 512, 512, 256, 256],
            paddings=[1, 1, 1,1,1,1],
            pool_type='max2d',
            pool_sizes=[2, 2, 2, 2, 2, 1],  # the one at the end means no pool
            pool_strides=[2, 2, 2,2,2,1],
            pool_paddings=[0, 0, 0,0,0,0],
            image_augmentation=True,
            image_augmentation_padding=4,
        )
    else:
        variant['cnn_params'] = dict(
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
        )

    variant['seed'] = args.seed
    ptu.set_gpu_mode(True)

    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None

    log_dir = setup_logger(args.name, variant=variant, base_log_dir=base_log_dir, snapshot_mode='gap_and_last', snapshot_gap=10,)
    variant['log_dir'] = log_dir
    experiment(variant)
