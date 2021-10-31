import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy, load_data_from_npy_mult
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector, CustomMDPPathCollector_EVAL, MdpPathCollector_Context

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql_context import CQLTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger

import argparse, os
import roboverse

# '/media/avi/data/Work/github/avisingh599/minibullet/data/'
#                   'oct6_Widow250DrawerGraspNeutral-v0_20K_save_all_noise_0.1'
#                   '_2020-10-06T19-37-26_100.npy'

# DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')

DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')
CUSTOM_LOG_DIR = '/nfs/kun1/users/asap7772/doodad-output/'


def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    expl_env = roboverse.make(variant['env'], transpose_image=True)
    action_dim = eval_env.action_space.low.size

    if variant['multi_bin']:
        eval_env.multi_tray = True
        expl_env.multi_tray = False

    context_dim=3

    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim+context_dim,
    )
    qf1 = ConcatCNN(**cnn_params)
    qf2 = ConcatCNN(**cnn_params)
    target_qf1 = ConcatCNN(**cnn_params)
    target_qf2 = ConcatCNN(**cnn_params)

    cnn_params.update(
        output_size=256,
        added_fc_input_size=context_dim,
        hidden_sizes=[1024, 512],
    )

    policy_obs_processor = CNN(**cnn_params)
    policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector_Context(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector_EVAL(
        eval_env,
        eval_policy,
    )

    observation_key = 'image'
    internal_keys = ['camera_orientation']
    
    if variant['transfer_multiview']:
        eval_env.multi_view = True
        eval_env.multi_view_type = variant['eval_multiview']

        if type(args.buffer) != list:
            replay_buffer = load_data_from_npy(variant, expl_env, observation_key, bin_change=variant['bin'], target_segment=variant['segment_type'], scale_rew=variant['trainer_kwargs']['with_lagrange'], internal_keys=internal_keys)
        else:
            p = [1] * len(args.buffer)
            bin_changes = [False] * len(args.buffer)
            target_segments = [None] * len(args.buffer)
            replay_buffer = load_data_from_npy_mult(variant, expl_env, observation_key, bin_changes=bin_changes, target_segments=target_segments, p = p, scale_rew=variant['trainer_kwargs']['with_lagrange'], internal_keys=internal_keys)
    else:
        replay_buffer = load_data_from_npy(variant, expl_env, observation_key, bin_change=variant['bin'], target_segment=variant['segment_type'], scale_rew=variant['trainer_kwargs']['with_lagrange'], internal_keys=internal_keys)

    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        dist_diff=variant['dist_diff'],
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
    parser.add_argument("--p", default=0.2, type=float)
    parser.add_argument('--segment_type', default='fixed_other', type = str)
    parser.add_argument('--eval_multiview', default='single', type = str)
    parser.add_argument('--larger_net', action="store_true", default=False)
    parser.add_argument('--dist_diff', action="store_true", default=False)

    args = parser.parse_args()
    variant['transfer'] = args.transfer
    variant['mixture'] = args.mixture
    variant['p'] = args.p
    variant['bin'] = args.bin_color
    variant['segment_type'] = args.segment_type
    
    variant['transfer_multiview'] = args.transfer_multiview
    variant['eval_multiview'] = args.eval_multiview
    variant['dist_diff'] = args.dist_diff

    if args.buffer.isnumeric():
        args.buffer = int(args.buffer)
    
    if variant['transfer_multiview']:
        if args.buffer == 1:
            args.buffer = '/nfs/kun1/users/asap7772/roboverse/data/single_grasp/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-30T00-07-28.npy'
        elif args.buffer == 2:
            args.buffer = '/nfs/kun1/users/asap7772/roboverse/data/train_grasp_orient/scripted_Widow250MultiObjectGraspTrain-v0_2021-01-07T00-41-13.npy'
        elif args.buffer == 3:
            args.buffer = '/nfs/kun1/users/asap7772/roboverse/data/val_grasp_orient/scripted_Widow250MultiObjectGraspTrain-v0_2021-01-07T00-41-30.npy'
        elif args.buffer == 4:
            args.buffer = '/nfs/kun1/users/asap7772/roboverse/data/all_grasp_orient/scripted_Widow250MultiObjectGraspTrain-v0_2021-01-07T00-41-35.npy'
        elif args.buffer == 5:
            rand = '/nfs/kun1/users/asap7772/roboverse/data/val_rand/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-30T00-07-07.npy'
            grasp = '/nfs/kun1/users/asap7772/roboverse/data/train_grasp/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-30T00-06-52.npy'
            args.buffer = [grasp, rand]
        elif args.buffer == 6:
            rand = '/nfs/kun1/users/asap7772/roboverse/data/train_rand/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-30T00-07-04.npy'
            grasp = '/nfs/kun1/users/asap7772/roboverse/data/val_grasp/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-30T00-06-55.npy'
            args.buffer = [grasp, rand]
        elif args.buffer == 7:
            rand1 = '/nfs/kun1/users/asap7772/roboverse/data/train_rand/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-30T00-07-04.npy'
            rand2 = '/nfs/kun1/users/asap7772/roboverse/data/val_rand/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-30T00-07-07.npy'
            grasp = '/nfs/kun1/users/asap7772/roboverse/data/val_grasp/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-30T00-06-55.npy'
            args.buffer = [rand1, rand2, grasp]

    elif args.buffer == 1:
        args.buffer = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-07T10-22-05.npy'
    elif args.buffer == 2:
        args.buffer = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-07T02-23-02.npy'
    elif args.buffer == 3:
        args.buffer = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-07T10-49-00.npy'
    elif args.buffer == 4:
        args.buffer = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-19T23-15-41.npy'
    elif args.buffer == 5:
        args.buffer = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-26T22-49-57.npy'
    elif args.buffer == 'test':
        args.buffer = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-14T09-15-08.npy'
    elif variant['transfer']:
        rand = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-26T22-49-57.npy'
        grasp = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-19T23-15-41.npy'
        args.buffer = [grasp, rand]
    


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

    setup_logger(args.name, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    experiment(variant)
