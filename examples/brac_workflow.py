import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector

import torch
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.brac import BRACTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger
import gym

import argparse, os
import roboverse
import numpy as np

DEFAULT_PRIOR_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
DEFAULT_TASK_BUFFER = ('/media/avi/data/Work/github/avisingh599/minibullet'
                        '/data/oct6_Widow250DrawerGraspNeutral-v0_20K_save_all'
                        '_noise_0.1_2020-10-06T19-37-26_100.npy')
CUSTOM_LOG_DIR = '/nfs/kun1/users/asap7772/doodad-output/'

def process_buffer(args):
    path = '/home/asap7772/cog_data/'
    buffers = []    
    home = os.path.expanduser("~")
    p_data_path =  os.path.join(home, 'prior_data/') if args.azure else '/nfs/kun1/users/asap7772/prior_data/' 
    ba = lambda x, p=args.prob, y=None: buffers.append((os.path.join(path, x),dict(p=p,alter_type=y,)))

    if args.buffer == 0:
        path = p_data_path
        ba('buffer1.npy', y='zero') 
        ba('buffer2.npy')

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
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
    )
    qf1 = ConcatCNN(**cnn_params)
    qf2 = ConcatCNN(**cnn_params)
    target_qf1 = ConcatCNN(**cnn_params)
    target_qf2 = ConcatCNN(**cnn_params)

    cnn_params.update(
        output_size=256,
        added_fc_input_size=0,
        hidden_sizes=[1024, 512],
    )

    policy_obs_processor = CNN(**cnn_params)

    behavior_policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
    )

    state_dict = torch.load(variant['behavior_path'])['policy_state_dict']
    behavior_policy.load_state_dict(state_dict)

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

    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )

    observation_key = 'image'
    replay_buffer = load_data_from_npy_chaining(
            variant,
            expl_env, 
            observation_key,
            duplicate=False,
            num_traj=variant['num_traj'],
            debug_scale_actions=False,
            debug_shift=False,
            scale_type=False,
            hist_state=False,
            num_hist=False,
        )

    # Translate 0/1 rewards to +4/+10 rewards.
    if variant['use_positive_rew']:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards * 6.0
            replay_buffer._rewards = replay_buffer._rewards + 4.0
        assert set(np.unique(replay_buffer._rewards)).issubset(
            set(6.0 * np.array([0, 1]) + 4.0))

    trainer = BRACTrainer(
        env=eval_env,
        policy=policy,
        behavior_policy=behavior_policy.to(ptu.device),
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        beta=variant['beta'],
        log_dir=variant['log_dir'],
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
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max-path-length", type=int, required=True)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--beta", default=1.0, type=float, help="Value of beta in BRAC")
    parser.add_argument("--use-positive-rew", action="store_true", default=False)
    parser.add_argument("--policy-eval-start", default=10000,type=int)
    parser.add_argument("--policy-lr", default=1e-4, type=float)
    parser.add_argument("--min-q-version", default=3, type=int,
                        help=("min_q_version = 3 (CQL(H)), "
                              "version = 2 (CQL(rho))"))
    parser.add_argument("--num-eval-per-epoch", type=int, default=5)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--buffer", default=0, type=int)
    parser.add_argument('--behavior_path', default='/nfs/kun1/users/asap7772/cog/data/behavior-bc/behavior_bc_2021_08_18_21_07_43_0000--s-0/model_pkl/200.pt', type=str)
    parser.add_argument('--num_traj', default=0, type=int)
    parser.add_argument("--prob", default=1, type=float)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--name", default='test', type=str)
    parser.add_argument("--azure", action='store_true')
    parser.add_argument('--eval_num', default=0, type=int)

    args = parser.parse_args()
    enable_gpus(args.gpu)
    variant['num_sample'] = args.eval_num
    variant['num_traj'] = args.num_traj
    variant['prob'] = args.prob
    variant['env'] = args.env
    variant['buffer'] = args.buffer
    variant['behavior_path'] = args.behavior_path
    variant['algorithm_kwargs']['max_path_length'] = args.max_path_length
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = \
        args.num_eval_per_epoch*args.max_path_length

    buffers = process_buffer(args)
    variant['buffer'] = buffers
    variant['bufferidx'] = args.buffer
    variant['beta'] = args.beta
    variant['behavior_path'] = args.behavior_path

    variant['prior_buffer'] = buffers[0]
    variant['task_buffer'] = buffers[1]

    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['discount'] = args.discount

    # Translate 0/1 rewards to +4/+10 rewards.
    variant['use_positive_rew'] = args.use_positive_rew
    variant['seed'] = args.seed

    ptu.set_gpu_mode(True)
    exp_prefix = 'cql-cog-{}'.format(args.env)
    if os.path.isdir(CUSTOM_LOG_DIR):
        base_log_dir = CUSTOM_LOG_DIR
    else:
        base_log_dir = None

    log_dir = setup_logger(args.name, variant=variant, base_log_dir=base_log_dir,
                 snapshot_mode='gap_and_last', snapshot_gap=10,)
    variant['log_dir'] = log_dir
    experiment(variant)