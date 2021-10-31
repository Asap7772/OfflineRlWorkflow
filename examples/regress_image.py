from traitlets.traitlets import default
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy, load_data_from_npy_mult, load_data_from_npy_split, load_data_from_npy_chaining
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector, CustomMDPPathCollector_EVAL

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.regress import RegressTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, ConcatCNNWrapperRegress
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger
from rlkit.envs.dummy_env import DummyEnv
import argparse, os
import roboverse
import torch

DEFAULT_BUFFER = ('/nfs/kun1/users/albert/minibullet_datasets/11270225_10k_grasp_Widow250MultiObjectOneGraspRandomBowlPositionTrain-v0_10K_save_all_noise_0.1_2020-11-27T02-24-16_9750.npy')
CUSTOM_LOG_DIR = '/nfs/kun1/users/asap7772/doodad-output/'


def experiment(variant):
    expl_env = eval_env = roboverse.make(variant['env'], transpose_image=True)
    action_dim = eval_env.action_space.low.size

    if variant['loc'] != '':
        parameters = torch.load(variant['loc'])
        
        cnn_params = variant['cnn_params']
        cnn_params.update(
            input_width=48,
            input_height=48,
            input_channels=3,
            output_size=1,
            added_fc_input_size=action_dim,
        )
        if variant['mcret'] or variant['bchead']:
            qf1 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'])
        elif variant['bottleneck']:
            qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
        else:
            qf1 = ConcatCNN(**cnn_params)
        qf1.load_state_dict(parameters['qf1_state_dict'])
        network = ConcatCNNWrapperRegress(qf1, 2, action_dim) # output size is 2

    else:
        cnn_params = variant['cnn_params']
        cnn_params.update(
            input_width=48,
            input_height=48,
            input_channels=3,
            output_size=3,
            added_fc_input_size=0,
        )
        network = ConcatCNN(**cnn_params)

    observation_key = 'image'
    replay_buffer = load_data_from_npy_chaining(variant, expl_env, observation_key, duplicate=variant['duplicate'], num_traj=variant['num_traj'])
    

    if variant['val']:
        if args.buffer in [5,6]:
            replay_buffer_val = load_data_from_npy_chaining_mult(
                variant, expl_env, observation_key)
        else:
            buffers = []
            ba = lambda x, p=args.prob, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))
            if args.buffer == 1 or args.buffer == 9001:
                ba('val_pick_2obj_Widow250PickTrayMult-v0_100_save_all_noise_0.1_2021-05-07T01-16-43_117.npy', p=args.prob,y='zero')
                ba('val_place_2obj_Widow250PlaceTrayMult-v0_100_save_all_noise_0.1_2021-05-07T01-16-48_108.npy', p=args.prob)
            
            old_pb, variant['prior_buffer'] = variant['prior_buffer'], buffers[0]
            old_tb, variant['task_buffer'] = variant['task_buffer'], buffers[1]
            old_nt, variant['num_traj'] = variant['num_traj'], 0

            replay_buffer_val = load_data_from_npy_chaining(
                variant, expl_env, observation_key, duplicate=variant['duplicate'], num_traj=variant['num_traj'])
            
            variant['prior_buffer'] = old_pb
            variant['task_buffer'] = old_tb
            variant['num_traj'] = old_nt
        print('validation')
    else:
        print('no validation')
        replay_buffer_val = None
    
    trainer = RegressTrainer(
        env=eval_env,
        network=network,
        alt_buffer=replay_buffer_val,
        **variant['trainer_kwargs']
    )

    if variant['dummy']:
        eval_env = DummyEnv()
    
    eval_policy = None
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector_EVAL(
        eval_env,
        eval_policy,
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
        algorithm="Regress",
        version="normal",
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=0,
            min_num_steps_before_training=0,
            max_path_length=30,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            network_lr=1E-4,
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='')
    parser.add_argument("--max-path-length", type=int, default='30')
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--network-lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--name", default='test', type=str)
    parser.add_argument("--loc", default='', type=str)
    parser.add_argument('--prob', default=1, type=float)
    parser.add_argument('--train_per', default='0.9', type = float)
    parser.add_argument('--mcret', action='store_true')
    parser.add_argument('--bchead', action='store_true')
    parser.add_argument('--bottleneck', action='store_true')
    parser.add_argument("--duplicate", action="store_true", default=False)
    parser.add_argument('--num_traj', default=0, type=int)
    parser.add_argument('--dummy', action='store_false', default=True)
    parser.add_argument('--val', action='store_false', default=True)

    args = parser.parse_args()
    variant['loc'] = args.loc
    variant['val'] = args.val
    variant['dummy'] =args.dummy
    variant['mcret'] = args.mcret
    variant['bchead'] = args.bchead
    variant['bottleneck'] = args.bottleneck
    variant['duplicate'] = args.duplicate
    variant['num_traj'] = args.num_traj
    
    if args.buffer.isnumeric():
        args.buffer = int(args.buffer)

    path = '/nfs/kun1/users/asap7772/cog_data/'
    buffers = []
    ba = lambda x, p=args.prob, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))
    if args.buffer == 1:
        path  = '/nfs/kun1/users/asap7772/prior_data/'
        ba('pick_2obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-43_5000.npy', p=args.prob,y='zero')
        ba('place_2obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-49_5000.npy', p=args.prob)
    variant['buffer'] = buffers
    variant['bufferidx'] = args.buffer

    if args.buffer in [5,6]:
        variant['prior_buffer'] = buffers[1:]
        variant['task_buffer'] = buffers[0]
    else:
        variant['prior_buffer'] = buffers[0]
        variant['task_buffer'] = buffers[1]

    
    enable_gpus(args.gpu)
    variant['env'] = args.env
    variant['buffer'] = args.buffer
    variant['trainer_kwargs']['network_lr'] = args.network_lr

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
