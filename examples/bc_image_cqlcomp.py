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

import argparse, os
import roboverse
from os.path import expanduser
from roboverse.envs.noise_wrapper import StochasticDynamicsWrapper

DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')
CUSTOM_LOG_DIR = '/home/asap7772/doodad-output'


def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    expl_env = roboverse.make(variant['env'], transpose_image=True)
    action_dim = eval_env.action_space.low.size

    if variant['multi_bin']:
        eval_env.multi_tray = True
        expl_env.multi_tray = False
    
    if variant['stoch_dynamics']:
        eval_env = StochasticDynamicsWrapper(eval_env, std=variant['stoch_dyn_std'])

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
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
    )

    cnn_params.update(
        output_size=256,
        added_fc_input_size=0,
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

    observation_key = 'image'

    if variant['transfer_multiview']:
        eval_env.multi_view = True
        eval_env.multi_view_type = variant['eval_multiview']

        if type(args.buffer) != list:
            replay_buffer = load_data_from_npy(variant, expl_env, observation_key, bin_change=variant['bin'], target_segment=variant['segment_type'], scale_rew=variant['trainer_kwargs']['with_lagrange'])
        else:
            p = [1] * len(args.buffer)
            bin_changes = [False] * len(args.buffer)
            target_segments = [None] * len(args.buffer)
            replay_buffer = load_data_from_npy_mult(variant, expl_env, observation_key, bin_changes=bin_changes, target_segments=target_segments, p = p, scale_rew=variant['trainer_kwargs']['with_lagrange'])

    elif variant['mixture']:
        p = [variant['p'], 1]
        bin_changes = [True, variant['bin']]
        target_segments = ['random_control', variant['segment_type']]
        replay_buffer = load_data_from_npy_mult(variant, expl_env, observation_key, bin_changes=bin_changes, target_segments=target_segments, p = p, scale_rew=variant['trainer_kwargs']['with_lagrange'])
    elif variant['transfer']:
        p = [variant['p'], 1]
        bin_changes = [False, variant['bin']]
        target_segments = [None, variant['segment_type']]
        replay_buffer = load_data_from_npy_mult(variant, expl_env, observation_key, bin_changes=bin_changes, target_segments=target_segments, p = p, scale_rew=variant['trainer_kwargs']['with_lagrange'])
    elif variant['chaining']:
        replay_buffer = load_data_from_npy_chaining(
            variant, expl_env, observation_key, duplicate=variant['duplicate'], num_traj=variant['num_traj'])
    else:
        replay_buffer = load_data_from_npy(variant, expl_env, observation_key, bin_change=variant['bin'], target_segment=variant['segment_type'], scale_rew=variant['trainer_kwargs']['with_lagrange'])

    if variant['val']:
        buffers = []
        ba = lambda x, p=args.prob, y=None: buffers.append((path + x, dict(p=p, alter_type=y, )))
        path =  os.path.join(home, 'prior_data/') if args.azure else '/nfs/kun1/users/aviral/prior_data/' 

        if args.buffer == 0:
            ba('val_expert_draweropen_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-54-53_1000.npy', y='zero')
            ba('val_expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-17_900.npy', y='zero')
        elif args.buffer == 1:
            ba('val_expert_draweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-54-57_1000.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-21_1000.npy', y='zero')
        elif args.buffer == 2:
            ba('val_expert_draweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-55-01_1000.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-26_1000.npy', y='zero')
        elif args.buffer == 3:
            ba('val_expert_draweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-55-04_900.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-30_900.npy', y='zero')
        elif args.buffer == 4:
            ba('val_expert_closedraweropen_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-55-40_1000.npy', y='zero')
            ba('val_expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-17_900.npy', y='zero')
        elif args.buffer == 5:
            ba('val_expert_closedraweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-55-44_1000.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-21_1000.npy', y='zero')
        elif args.buffer == 6:
            ba('val_expert_closedraweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-55-47_1000.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-26_1000.npy', y='zero')
        elif args.buffer == 7:
            ba('val_expert_closedraweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-55-52_1000.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-30_900.npy', y='zero')
        elif args.buffer == 8:
            ba('val_expert_draweropen_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-54-53_1000.npy', y='zero')
            ba('val_expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-17_900.npy', y='zero')
        elif args.buffer == 9:
            ba('val_expert_draweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-54-57_1000.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-21_1000.npy', y='zero')
        elif args.buffer == 10:
            ba('val_expert_draweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-55-01_1000.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-26_1000.npy', y='zero')
        elif args.buffer == 11:
            ba('val_expert_draweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-55-04_900.npy', y='zero')
            ba('val_expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-57-30_900.npy', y='zero')

        old_pb, variant['prior_buffer'] = variant['prior_buffer'], buffers[0]
        old_tb, variant['task_buffer'] = variant['task_buffer'], buffers[1]
        old_nt, variant['num_traj'] = variant['num_traj'], 0

        replay_buffer_val = load_data_from_npy_chaining(
            variant, expl_env, observation_key, duplicate=variant['duplicate'], num_traj=variant['num_traj'])

        variant['prior_buffer'] = old_pb
        variant['task_buffer'] = old_tb
        variant['num_traj'] = old_nt
        print('validation buffer loaded')
    else:
        replay_buffer_val = None
        print('no validation buffer')


    trainer = BCTrainer(
        env=eval_env,
        policy=policy,
        #qf1=qf1,
        #qf2=qf2,
        #target_qf1=target_qf1,
        #target_qf2=target_qf2,
        validation=variant['val'],
        validation_buffer=replay_buffer_val,
        dist_diff=variant['dist_diff'],
        log_dir=variant['log_dir'],
        variant_dict=variant,
        bc_cql_comp=True,
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
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max-path-length", type=int, required=True)
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
    parser.add_argument("--chaining", action="store_false", default=True)
    parser.add_argument("--p", default=0.2, type=float)
    parser.add_argument("--prob", default=1.0, type=float)
    parser.add_argument('--segment_type', default='fixed_other', type = str)
    parser.add_argument('--eval_multiview', default='single', type = str)
    parser.add_argument('--dist_diff', action="store_true", default=False)

    parser.add_argument('--larger_net', action="store_true", default=False)
    
    # Stephen added

    parser.add_argument('--val', action="store_false", default=True)

    parser.add_argument('--deeper_net', action="store_true", default=False)
    parser.add_argument('--vqvae_enc', action="store_true", default=False)
    parser.add_argument('--duplicate', action="store_true", default=False)
    parser.add_argument('--num_traj', default=0, type=int)

    parser.add_argument('--azure', action='store_true')
    parser.add_argument('--stochastic_dynamics', action='store_true')
    parser.add_argument('--stochastic_noise', type=float, default=0.1)


    args = parser.parse_args()
    variant['transfer'] = args.transfer
    variant['mixture'] = args.mixture
    variant['chaining'] = args.chaining
    variant['p'] = args.p
    variant['bin'] = args.bin_color
    variant['segment_type'] = args.segment_type

    variant['stoch_dynamics'] = args.stochastic_dynamics
    variant['stoch_dyn_std'] = args.stochastic_noise

    variant['val'] = args.val

    variant['transfer_multiview'] = args.transfer_multiview
    variant['eval_multiview'] = args.eval_multiview
    variant['dist_diff'] = args.dist_diff

    variant['deeper_net'] = args.deeper_net
    variant['vqvae_enc'] = args.vqvae_enc
    variant['duplicate'] = args.duplicate
    variant['num_traj'] = args.num_traj

    if args.buffer.isnumeric():
        args.buffer = int(args.buffer)
    
    home = expanduser("~")
    path =  os.path.join(home, 'prior_data/') if args.azure else '/nfs/kun1/users/aviral/prior_data/' 
    buffers = []
    ba = lambda x, p=args.prob, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))

    if args.buffer == 0:
        ba('expert_draweropen_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-42-19_900.npy', y='zero')
        ba('expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-09_1000.npy', y='zero')
    elif args.buffer == 1:	
        ba('expert_draweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-42-22_1000.npy', y='zero')
        ba('expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-14_1000.npy', y='zero')
    elif args.buffer == 2:
        ba('expert_draweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-42-25_900.npy', y='zero')
        ba('expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-18_800.npy', y='zero')
    elif args.buffer == 3:
        ba('expert_draweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-42-29_800.npy', y='zero')
        ba('expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-22_600.npy', y='zero')
    elif args.buffer == 4:
        ba('expert_draweropen_nosaveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-39-01_900.npy', y='zero')
        ba('expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-09_1000.npy', y='zero')
    elif args.buffer == 5:
        ba('expert_draweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-39-05_1000.npy', y='zero')
        ba('expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-14_1000.npy', y='zero')
    elif args.buffer == 6:
        ba('expert_draweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-39-09_1000.npy', y='zero')
        ba('expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-18_800.npy', y='zero')
    elif args.buffer == 7:
        ba('expert_draweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-39-13_1000.npy', y='zero')
        ba('expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-22_600.npy', y='zero')
    elif args.buffer == 8:
        ba('expert_closedraweropen_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-40-58_900.npy', y='zero')
        ba('expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-09_1000.npy', y='zero')
    elif args.buffer == 9:
        ba('expert_closedraweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-41-01_1000.npy', y='zero')
        ba('expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-14_1000.npy', y='zero')
    elif args.buffer == 10:
        ba('expert_closedraweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-41-05_800.npy', y='zero')
        ba('expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-18_800.npy', y='zero')
    elif args.buffer == 11:
        ba('expert_closedraweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-41-08_900.npy', y='zero')
        ba('expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-22_600.npy', y='zero')
    
    variant['buffer'] = buffers
    variant['bufferidx'] = args.buffer

    if variant['buffer'] is not None:
        variant['prior_buffer'] = buffers[0]
        variant['task_buffer'] = buffers[1]

    enable_gpus(args.gpu)
    variant['env'] = args.env
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
    exp_prefix = 'cql-{}'.format(args.env)

    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None

    log_dir = setup_logger(args.name, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    variant['log_dir'] = log_dir
    experiment(variant)
