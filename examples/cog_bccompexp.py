from git.index import typ
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.load_buffer import load_data_from_npy_chaining,load_data_from_npy_chaining_mult
from rlkit.samplers.data_collector import MdpPathCollector, \
    CustomMDPPathCollector

from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.sac.cql_montecarlo import CQLMCTrainer
from rlkit.torch.sac.cql_bchead import CQLBCTrainer
from rlkit.torch.sac.cql_single import CQLSingleTrainer
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, VQVAEEncoderConcatCNN, \
    ConcatBottleneckVQVAECNN, VQVAEEncoderCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.util.video import VideoSaveFunction
from rlkit.launchers.launcher_util import setup_logger

import argparse, os
import roboverse
from roboverse.envs.noise_wrapper import StochasticDynamicsWrapper
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
CUSTOM_LOG_DIR = '/home/stian/doodad-output'


def experiment(variant):
    eval_env = roboverse.make(variant['env'], transpose_image=True)
    if variant['num_sample'] != 0:
        eval_env.num_obj_sample=variant['num_sample']

    if variant['stoch_dynamics']:
        eval_env = StochasticDynamicsWrapper(eval_env, std=variant['stoch_dyn_std'])

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


    if variant['spectral_norm_conv']:
        cnn_params.update(
            spectral_norm_conv=True,
        )
    if variant['spectral_norm_fc']:
        cnn_params.update(
            spectral_norm_fc=True,
        )

    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
        normalize_conv_activation=variant['normalize_conv_activation']
    )
    if variant['vqvae_enc']:
        if variant['bottleneck']:
            qf1 = ConcatBottleneckVQVAECNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],
                                      deterministic=variant['deterministic_bottleneck'],
                                      spectral_norm_conv=cnn_params['spectral_norm_conv'],
                                      spectral_norm_fc=cnn_params['spectral_norm_fc'])
            qf2 = ConcatBottleneckVQVAECNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],
                                      deterministic=variant['deterministic_bottleneck'],
                                      spectral_norm_conv = cnn_params['spectral_norm_conv'],
                                      spectral_norm_fc = cnn_params['spectral_norm_fc'])

            if variant['share_encoder']:
                print('sharing encoder weights between QF1 and QF2!')
                qf2.encoder = qf1.encoder

            target_qf1 = ConcatBottleneckVQVAECNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],
                                             deterministic=variant['deterministic_bottleneck'],
                                             spectral_norm_conv=cnn_params['spectral_norm_conv'],
                                             spectral_norm_fc=cnn_params['spectral_norm_fc'])
            target_qf2 = ConcatBottleneckVQVAECNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],
                                             deterministic=variant['deterministic_bottleneck'],
                                             spectral_norm_conv=cnn_params['spectral_norm_conv'],
                                             spectral_norm_fc=cnn_params['spectral_norm_fc'])
        else:
            qf1 = VQVAEEncoderConcatCNN(**cnn_params)
            qf2 = VQVAEEncoderConcatCNN(**cnn_params)
            if variant['share_encoder']:
                print('sharing encoder weights between QF1 and QF2!')
                del qf2.encoder
                qf2.encoder = qf1.encoder
            target_qf1 = VQVAEEncoderConcatCNN(**cnn_params)
            target_qf2 = VQVAEEncoderConcatCNN(**cnn_params)

    else:
        if variant['mcret'] or variant['bchead']:
            qf1 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'])
            qf2 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'])
            target_qf1 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'])
            target_qf2 = TwoHeadCNN(action_dim, deterministic= not variant['bottleneck'], bottleneck_dim=variant['bottleneck_dim'])
            if variant['share_encoder']:
                raise NotImplementedError

        elif variant['bottleneck']:
            qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
            qf2 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
            if variant['share_encoder']:
                raise NotImplementedError
            target_qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
            target_qf2 = ConcatBottleneckCNN(action_dim, bottleneck_dim=variant['bottleneck_dim'],deterministic=variant['deterministic_bottleneck'])
        else:
            qf1 = ConcatCNN(**cnn_params)
            qf2 = ConcatCNN(**cnn_params)
            if variant['share_encoder']:
                raise NotImplementedError
            target_qf1 = ConcatCNN(**cnn_params)
            target_qf2 = ConcatCNN(**cnn_params)

    target_qf1.load_state_dict(qf1.state_dict())
    target_qf2.load_state_dict(qf2.state_dict())

    if variant['vqvae_policy']:
        if variant['share_encoder']:
            print('sharing encoder weights between QF and Policy with VQVAE Encoder')
            policy_obs_processor = qf1.encoder
            cnn_params.update(
                output_size=qf1.get_conv_output_size(),
            )
        else:
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
    if args.buffer in [5,6]:
        replay_buffer = load_data_from_npy_chaining_mult(
            variant, expl_env, observation_key)
    else:
        replay_buffer = load_data_from_npy_chaining(
            variant, expl_env, observation_key, duplicate=variant['duplicate'], num_traj=variant['num_traj'])

    if variant['val']:
        if args.buffer in [5,6]:
            replay_buffer_val = load_data_from_npy_chaining_mult(
                variant, expl_env, observation_key)
        else:
            buffers = [
            ]
            ba = lambda x, p=args.prob, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))
            if args.buffer == 30:
                path = p_data_path
                ba('val_pick_2obj_Widow250PickTrayMult-v0_100_save_all_noise_0.1_2021-05-07T01-16-43_117.npy', p=args.prob,y='zero')
                ba('val_place_2obj_Widow250PlaceTrayMult-v0_100_save_all_noise_0.1_2021-05-07T01-16-48_108.npy', p=args.prob)
            if args.buffer == 32 or args.buffer == 9001:
                path = p_data_path
                ba('val_pick_2obj_Widow250PickTrayMult-v0_100_save_all_noise_0.1_2021-05-07T01-16-43_117.npy', p=args.prob,y='zero')
                ba('val_place_2obj_Widow250PlaceTrayMult-v0_100_save_all_noise_0.1_2021-05-07T01-16-48_108.npy', p=args.prob)
            elif args.buffer == 35:
                path = p_data_path
                ba('val_pick_35_Widow250PickTrayMult-v0_100_save_all_noise_0.1_2021-06-14T21-52-13_100.npy',
                   p=args.prob, y='zero')
                ba('val_place_35_Widow250PlaceTrayMult-v0_100_save_all_noise_0.1_2021-06-14T21-50-14_100.npy',
                   p=args.prob)
            elif args.buffer == 36:
                path = p_data_path
                ba('val_pick_20obj_Widow250PickTrayMult-v0_100_save_all_noise_0.1_2021-05-07T01-16-53_114.npy',
                   p=args.prob, y='zero')
                ba('val_place_20obj_Widow250PlaceTrayMult-v0_100_save_all_noise_0.1_2021-05-07T01-16-58_90.npy',
                   p=args.prob)

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

    # Translate 0/1 rewards to +4/+10 rewards.
    if variant['use_positive_rew']:
        if set(np.unique(replay_buffer._rewards)).issubset({0, 1}):
            replay_buffer._rewards = replay_buffer._rewards * 6.0
            replay_buffer._rewards = replay_buffer._rewards + 4.0
        assert set(np.unique(replay_buffer._rewards)).issubset(
            set(6.0 * np.array([0, 1]) + 4.0))

    if variant['mcret']:
        trainer = CQLMCTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            bottleneck=variant['bottleneck'],
            bottleneck_const=variant['bottleneck_const'],
            bottleneck_lagrange=variant['bottleneck_lagrange'],
            dr3=variant['dr3'],
            dr3_feat=variant['dr3_feat'],
            dr3_weight=variant['dr3_weight'],
            log_dir = variant['log_dir'],
            wand_b=not variant['debug'],
            only_bottleneck = variant['only_bottleneck'],
            variant_dict=variant,
            gamma=variant['gamma'],
            **variant['trainer_kwargs']
        )
    elif variant['bchead']:
        trainer = CQLBCTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            bottleneck=variant['bottleneck'],
            bottleneck_const=variant['bottleneck_const'],
            bottleneck_lagrange=variant['bottleneck_lagrange'],
            dr3=variant['dr3'],
            dr3_feat=variant['dr3_feat'],
            dr3_weight=variant['dr3_weight'],
            log_dir = variant['log_dir'],
            wand_b=not variant['debug'],
            only_bottleneck = variant['only_bottleneck'],
            variant_dict=variant,
            gamma=variant['gamma'],
            **variant['trainer_kwargs']
        )
    elif variant['singleQ']:
        trainer = CQLSingleTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            target_qf1=target_qf1,
            bottleneck=variant['bottleneck'],
            bottleneck_const=variant['bottleneck_const'],
            bottleneck_lagrange=variant['bottleneck_lagrange'],
            dr3=variant['dr3'],
            dr3_feat=variant['dr3_feat'],
            dr3_weight=variant['dr3_weight'],
            only_bottleneck = variant['only_bottleneck'],
            log_dir = variant['log_dir'],
            wand_b=not variant['debug'],
            variant_dict=variant,
            validation=variant['val'],
            validation_buffer=replay_buffer_val,
            squared=variant['squared'],
            **variant['trainer_kwargs']
        )
        del qf2, target_qf2
        import torch; torch.cuda.empty_cache()
    else:
        trainer = CQLTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            bottleneck=variant['bottleneck'],
            bottleneck_const=variant['bottleneck_const'],
            bottleneck_lagrange=variant['bottleneck_lagrange'],
            dr3=variant['dr3'],
            dr3_feat=variant['dr3_feat'],
            dr3_weight=variant['dr3_weight'],
            only_bottleneck = variant['only_bottleneck'],
            log_dir = variant['log_dir'],
            wand_b=not variant['debug'],
            variant_dict=variant,
            validation=variant['val'],
            validation_buffer=replay_buffer_val,
            regularization=variant['regularization'],
            regularization_type=variant['regularization_type'],
            regularization_const=variant['regularization_const'],
            squared=variant['squared'],
            modify_grad=variant['modify_grad'],
            modify_type=variant['modify_type'],
            orthogonalize_grads=variant['orthogonalize_grads'],
            no_td=variant['no_td'],
            no_data_qval=variant['no_data_qval'],
            shifting=variant['shifting'],
            rew_regress=variant['rew_regress'],
            clip_grad_val=variant['clip_grad_val'],
            modify_func=variant['modify_func'],
            modify_func_type=variant['modify_func_type'],
            modify_func_const=variant['modify_func_const'],
            moving_mfconst=variant['moving_mfconst'],
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
            spectral_norm_conv=False,
            spectral_norm_fc=False,
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--max-path-length", type=int, required=True)
    parser.add_argument("--bottleneck", action='store_true')
    parser.add_argument('--bottleneck_const', type=float, default=0.5)
    parser.add_argument('--bottleneck_dim', type=int, default=16)
    parser.add_argument('--bottleneck_lagrange', action='store_true')
    parser.add_argument("--deterministic_bottleneck", action="store_true", default=False)
    parser.add_argument("--only_bottleneck", action="store_true", default=False)
    parser.add_argument("--mcret", action='store_true')
    parser.add_argument("--bchead", action='store_true')
    parser.add_argument("--prior-buffer", type=str, default=DEFAULT_PRIOR_BUFFER)
    parser.add_argument("--task-buffer", type=str, default=DEFAULT_TASK_BUFFER)
    parser.add_argument("--buffer", type=str, default=DEFAULT_BUFFER)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--min-q-weight", default=1.0, type=float,
                        help="Value of alpha in CQL")
    parser.add_argument("--use-lagrange", action="store_true", default=False)
    parser.add_argument("--lagrange-thresh", default=5.0, type=float,
                        help="Value of tau, used with --use-lagrange")
    parser.add_argument("--use-positive-rew", action="store_true", default=False)
    parser.add_argument("--duplicate", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
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
    parser.add_argument("--prob", default=1, type=float)
    parser.add_argument("--old_prior_prob", default=0, type=float)
    parser.add_argument('--gamma', default=1, type=float)
    parser.add_argument('--num_traj', default=0, type=int)
    parser.add_argument('--eval_num', default=0, type=int)
    parser.add_argument("--name", default='test', type=str)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument('--only_one', action='store_true')
    parser.add_argument("--squared", action="store_true", default=False)
    parser.add_argument("--azure", action="store_true", default=False)
    parser.add_argument("--bigger_net", action="store_true", default=False)
    parser.add_argument("--deeper_net", action="store_true", default=False)
    parser.add_argument("--vqvae_enc", action="store_true", default=False)
    parser.add_argument("--vqvae_policy", action="store_true", default=False)
    parser.add_argument("--share_encoder", action="store_true", default=False)
    parser.add_argument("--spectral_norm_conv", action="store_true", default=False)
    parser.add_argument("--spectral_norm_fc", action="store_true", default=False)
    parser.add_argument("--dr3", action="store_true", default=False)
    parser.add_argument("--dr3_feat", action="store_true", default=False)
    parser.add_argument("--dr3_weight", default=0.001, type=float)
    parser.add_argument("--eval_every_n", default=1, type=int)
    parser.add_argument('--singleQ', action='store_true')
    parser.add_argument('--smaller_net', action='store_true')
    parser.add_argument('--regularization', action='store_true')
    parser.add_argument('--regularization_type', type=str, default='l1')
    parser.add_argument('--regularization_const', type=float, default=1)
    parser.add_argument('--normalize_conv_activation', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--modify_grad', action='store_true')
    parser.add_argument('--modify_type', default=None, type=str)
    parser.add_argument('--orthogonalize_grads', action='store_true')
    parser.add_argument('--clip_targets', action='store_true')
    parser.add_argument('--target_clip_val', type=float, default=-250)

    parser.add_argument('--no_td', action='store_true')
    parser.add_argument('--no_data_qval', action='store_true')
    parser.add_argument('--shifting', action='store_true')
    parser.add_argument('--rew_regress', action='store_true')
    parser.add_argument('--clip_grad_val', type=float, default=10)

    parser.add_argument('--modify_func', action='store_true')
    parser.add_argument('--modify_func_type', default=None, type=str)
    parser.add_argument('--modify_func_const', default=1, type=float)  
    parser.add_argument('--moving_mfconst', action='store_true')
    
    parser.add_argument('--stochastic_dynamics', action='store_true')
    parser.add_argument('--stochastic_noise', type=float, default=0.1)

    args = parser.parse_args()
    enable_gpus(args.gpu)

    variant['modify_func'] = args.modify_func
    variant['modify_func_type'] = args.modify_func_type
    variant['modify_func_const'] = args.modify_func_const
    variant['moving_mfconst'] = args.moving_mfconst
    variant['stoch_dynamics'] = args.stochastic_dynamics
    variant['stoch_dyn_std'] = args.stochastic_noise

    variant['no_td'] = args.no_td
    variant['no_data_qval'] = args.no_data_qval
    variant['shifting'] = args.shifting
    variant['rew_regress'] = args.rew_regress
    variant['clip_grad_val'] = args.clip_grad_val

    variant['modify_grad'] = args.modify_grad
    variant['modify_type'] = args.modify_type
    variant['orthogonalize_grads'] = args.orthogonalize_grads

    variant['clip_targets'] = args.clip_targets
    variant['target_clip_val'] = args.target_clip_val
    
    variant['smaller_net'] = args.smaller_net
    variant['regularization'] = args.regularization
    variant['regularization_type'] = args.regularization_type
    variant['regularization_const'] = args.regularization_const

    variant['dropout'] = args.dropout
    variant['dropout_prob'] = args.dropout_prob

    variant['trainer_kwargs']['discount'] = args.discount
    variant['squared'] = args.squared
    variant['bigger_net'] = args.bigger_net
    variant['deeper_net'] = args.deeper_net
    variant['vqvae_enc'] = args.vqvae_enc
    variant['vqvae_policy'] = args.vqvae_policy
    variant['share_encoder'] = args.share_encoder
    variant['singleQ'] = args.singleQ
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
    variant['gamma'] = args.gamma
    variant['num_traj'] = args.num_traj
    variant['num_sample'] = args.eval_num
    variant['trainer_kwargs']['discount'] = args.discount
    
    variant['debug'] = False
    if args.buffer.isnumeric():
        args.buffer = int(args.buffer)
        
        home = expanduser("~")
        
        path =  os.path.join(home, 'prior_data/') if args.azure else '/nfs/kun1/users/aviral/prior_data/' 
        
        buffers = []
        ba = lambda x, p=args.prob, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))
        
        # =============================================================================
        if args.buffer == 0:
            print('Expert OG')
            print('Noise 0')
            print('Env Noise 0')

            ba('expert_draweropen_saveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-41-40_1000.npy', y='zero')
            ba('expert_grasp_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-44_800.npy', y='zero')
        
        elif args.buffer == 1:
            print('Expert OG')
            print('Noise 0')
            print('Env Noise 0.1')

            ba('expert_draweropen_stocdynam0.1_saveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-41-48_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.1_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-53_1000.npy', y='zero')

        elif args.buffer == 2:
            print('Expert OG')
            print('Noise 0')
            print('Env Noise 0.2')

            ba('expert_draweropen_stocdynam0.2_saveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-41-52_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.2_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-57_900.npy', y='zero')
        
        elif args.buffer == 3:
            print('Expert OG')
            print('Noise 0')
            print('Env Noise 0.3')

            ba('expert_draweropen_stocdynam0.3_saveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-41-55_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.3_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-46-00_1000.npy', y='zero')
        
        # =============================================================================

        if args.buffer == 4:
            print('Expert PPO')
            print('Noise 0')
            print('Env Noise 0')

            ba('expert_draweropen_saveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-36-43_1000.npy', y='zero')
            ba('expert_grasp_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-44_800.npy', y='zero')
        
        elif args.buffer == 5:
            print('Expert PPO')
            print('Noise 0')
            print('Env Noise 0.1')

            ba('expert_draweropen_stocdynam0.1_saveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-36-58_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.1_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-53_1000.npy', y='zero')

        elif args.buffer == 6:
            print('Expert PPO')
            print('Noise 0')
            print('Env Noise 0.2')

            ba('expert_draweropen_stocdynam0.2_saveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-37-03_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.2_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-57_900.npy', y='zero')
        
        elif args.buffer == 7:
            print('Expert PPO')
            print('Noise 0')
            print('Env Noise 0.3')

            ba('expert_draweropen_stocdynam0.3_saveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-37-07_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.3_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-46-00_1000.npy', y='zero')
        # =============================================================================
        if args.buffer == 8:
            print('Expert COG')
            print('Noise 0')
            print('Env Noise 0')

            ba('expert_closedraweropen_saveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-40-23_1000.npy', y='zero')
            ba('expert_grasp_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-44_800.npy', y='zero')
        
        elif args.buffer == 9:
            print('Expert COG')
            print('Noise 0')
            print('Env Noise 0.1')

            ba('expert_closedraweropen_stocdynam0.1_saveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-40-27_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.1_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-53_1000.npy', y='zero')
        
        elif args.buffer == 10:
            print('Expert COG')
            print('Noise 0')
            print('Env Noise 0.2')

            ba('expert_closedraweropen_stocdynam0.2_saveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-40-33_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.2_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-45-57_900.npy', y='zero')
        
        elif args.buffer == 11:
            print('Expert COG')
            print('Noise 0')
            print('Env Noise 0.3')

            ba('expert_closedraweropen_stocdynam0.3_saveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-40-36_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.3_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.0_2021-09-18T11-46-00_1000.npy', y='zero')
        # =============================================================================
        if args.buffer == 12:
            print('Suboptimal OG')
            print('Noise 0.1')
            print('Env Noise 0.1')

            ba('suboptimal_draweropen_saveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-42-03_1000.npy', y='zero')
            ba('suboptimal_grasp_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-47_1000.npy', y='zero')
        
        elif args.buffer == 13:
            print('Suboptimal OG')
            print('Noise 0.1')
            print('Env Noise 0.1')

            ba('suboptimal_draweropen_stocdynam0.1_saveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-42-06_1000.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.1_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-50_1000.npy', y='zero')

        elif args.buffer == 14:
            print('Suboptimal OG')
            print('Noise 0')
            print('Env Noise 0.2')

            ba('suboptimal_draweropen_stocdynam0.2_saveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-42-10_1000.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.2_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-53_1000.npy', y='zero')
        
        elif args.buffer == 15:
            print('Suboptimal OG')
            print('Noise 0.1')
            print('Env Noise 0.3')

            ba('suboptimal_draweropen_stocdynam0.3_saveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-42-13_900.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.3_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-57_1000.npy', y='zero')
        
        # =============================================================================

        if args.buffer == 16:
            print('Suboptimal PPO')
            print('Noise 0.1')
            print('Env Noise 0')

            ba('suboptimal_draweropen_saveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-37-31_1000.npy', y='zero')
            ba('suboptimal_grasp_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-47_1000.npy', y='zero')
        
        elif args.buffer == 17:
            print('Suboptimal PPO')
            print('Noise 0.1')
            print('Env Noise 0.1')

            ba('suboptimal_draweropen_stocdynam0.1_saveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-37-37_900.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.1_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-50_1000.npy', y='zero')

        elif args.buffer == 18:
            print('Suboptimal PPO')
            print('Noise 0.1')
            print('Env Noise 0.2')

            ba('suboptimal_draweropen_stocdynam0.2_saveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-37-41_1000.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.2_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-53_1000.npy', y='zero')
        
        elif args.buffer == 19:
            print('Suboptimal PPO')
            print('Noise 0.1')
            print('Env Noise 0.3')

            ba('suboptimal_draweropen_stocdynam0.3_saveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-37-45_1000.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.3_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-57_1000.npy', y='zero')
        # =============================================================================
        if args.buffer == 20:
            print('Suboptimal COG')
            print('Noise 0.1')
            print('Env Noise 0')

            ba('suboptimal_closedraweropen_saveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-40-41_1000.npy', y='zero')
            ba('suboptimal_grasp_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-47_1000.npy', y='zero')
        
        elif args.buffer == 21:
            print('Suboptimal COG')
            print('Noise 0.1')
            print('Env Noise 0.1')

            ba('suboptimal_closedraweropen_stocdynam0.1_saveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-40-45_1000.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.1_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-50_1000.npy', y='zero')
        
        elif args.buffer == 22:
            print('Suboptimal COG')
            print('Noise 0.1')
            print('Env Noise 0.2')

            ba('suboptimal_closedraweropen_stocdynam0.2_saveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-40-49_900.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.2_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-53_1000.npy', y='zero')
        
        elif args.buffer == 23:
            print('Suboptimal COG')
            print('Noise 0.1')
            print('Env Noise 0.3')

            ba('suboptimal_closedraweropen_stocdynam0.3_saveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-40-53_900.npy', y='zero')
            ba('suboptimal_grasp_stocdynam0.3_saveall_Widow250DoubleDrawerGraspNeutral-v0_1000_save_all_noise_0.1_2021-09-18T11-49-57_1000.npy', y='zero')
        # ======================
        if args.buffer == 24:
            ba('expert_draweropen_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-42-19_900.npy', y='zero')
            ba('expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-09_1000.npy', y='zero')
        elif args.buffer == 25:	
            ba('expert_draweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-42-22_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-14_1000.npy', y='zero')
        elif args.buffer == 26:
            ba('expert_draweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-42-25_900.npy', y='zero')
            ba('expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-18_800.npy', y='zero')
        elif args.buffer == 27:
            ba('expert_draweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-42-29_800.npy', y='zero')
            ba('expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-22_600.npy', y='zero')
        # ======================
        elif args.buffer == 28:
            ba('expert_draweropen_nosaveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-39-01_900.npy', y='zero')
            ba('expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-09_1000.npy', y='zero')
        elif args.buffer == 29:
            ba('expert_draweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-39-05_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-14_1000.npy', y='zero')
        elif args.buffer == 30:
            ba('expert_draweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-39-09_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-18_800.npy', y='zero')
        elif args.buffer == 31:
            ba('expert_draweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerPickPlaceOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-39-13_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-22_600.npy', y='zero')
        # ======================
        elif args.buffer == 32:
            ba('expert_closedraweropen_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-40-58_900.npy', y='zero')
            ba('expert_grasp_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-09_1000.npy', y='zero')
        elif args.buffer == 33:
            ba('expert_closedraweropen_stocdynam0.1_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-41-01_1000.npy', y='zero')
            ba('expert_grasp_stocdynam0.1_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-14_1000.npy', y='zero')
        elif args.buffer == 34:
            ba('expert_closedraweropen_stocdynam0.2_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-41-05_800.npy', y='zero')
            ba('expert_grasp_stocdynam0.2_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-18_800.npy', y='zero')
        elif args.buffer == 35:
            ba('expert_closedraweropen_stocdynam0.3_nosaveall_Widow250DoubleDrawerCloseOpenGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-41-08_900.npy', y='zero')
            ba('expert_grasp_stocdynam0.3_nosaveall_Widow250DoubleDrawerGraspNeutral-v0_1000_noise_0.0_2021-09-18T11-53-22_600.npy', y='zero')
        
        variant['buffer'] = buffers
        variant['bufferidx'] = args.buffer
    else:
        variant['buffer'] = None
    
    if variant['buffer'] is not None:
        if args.buffer in [5,6]:
            variant['prior_buffer'] = buffers[1:]
            variant['task_buffer'] = buffers[0]
        else:
            variant['prior_buffer'] = buffers[0]
            variant['task_buffer'] = buffers[1]

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
    variant['mcret'] = args.mcret
    variant['bchead'] = args.bchead

    # Translate 0/1 rewards to +4/+10 rewards.
    variant['use_positive_rew'] = args.use_positive_rew
    variant['seed'] = args.seed

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
