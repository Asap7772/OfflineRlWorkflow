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

    args = parser.parse_args()
    enable_gpus(args.gpu)

    variant['modify_func'] = args.modify_func
    variant['modify_func_type'] = args.modify_func_type
    variant['modify_func_const'] = args.modify_func_const
    variant['moving_mfconst'] = args.moving_mfconst

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
        p_data_path =  os.path.join(home, 'prior_data/') if args.azure else '/nfs/kun1/users/asap7772/prior_data/' 
        # p_data_path = '/home/stephentian/prior_data/'
        
        path = '/home/asap7772/cog_data/' if args.azure else '/nfs/kun1/users/asap7772/cog_data/'
        # path = '/home/stian/cog_data/'
        buffers = []
        ba = lambda x, p=args.prob, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))
        if args.buffer == 0:
            ba('closed_drawer_prior.npy',y='zero')
            path = p_data_path
            ba('task_singleneut_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-25T22-52-59_9750.npy')
        elif args.buffer == 1:
            ba('closed_drawer_prior.npy',y='zero')
            ba('drawer_task.npy')
        elif args.buffer == 2:
            ba('closed_drawer_prior.npy',y='zero')
            ba('drawer_task.npy',y='noise')
        elif args.buffer == 3:
            ba('closed_drawer_prior.npy',y='noise')
            ba('drawer_task.npy',y='zero')
        elif args.buffer == 4:
            ba('closed_drawer_prior.npy',y='noise')
            ba('drawer_task.npy',y='noise')
        elif args.buffer == 5:
            ba('drawer_task.npy')
            if args.old_prior_prob > 0:
                ba('closed_drawer_prior.npy',y='zero',p=args.old_prior_prob)
            path = p_data_path
            ba('grasp_newenv_Widow250DoubleDrawerOpenGraspNeutral-v0_20K_save_all_noise_0.1_2021-03-18T01-36-52_20000.npy',y='zero')
            ba('pickplace_newenv_Widow250PickPlaceMultiObjectMultiContainerTrain-v0_20K_save_all_noise_0.1_2021-03-18T01-38-58_19500.npy',y='zero')
            ba('drawer_newenv_Widow250DoubleDrawerOpenGraspNeutral-v0_20K_save_all_noise_0.1_2021-03-18T01-37-08_19500.npy', y='zero')
        elif args.buffer == 6:
            path = p_data_path
            ba('task_multneut_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-25T22-53-21_9250.npy')
            if args.old_prior_prob > 0:
                path = '/nfs/kun1/users/asap7772/cog_data/'
                ba('closed_drawer_prior.npy',y='zero',p=args.old_prior_prob)
                ba('drawer_task.npy',y='noise')
                path = p_data_path
            ba('grasp_multneut_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-24T01-17-30_10000.npy', y='zero')
            ba('double_drawer_multneut_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-03-24T01-19-23_9750.npy', y='zero')
        elif args.buffer == 7:
            path = p_data_path
            ba('pick_Widow250PickTray-v0_10K_save_all_noise_0.1_2021-04-03T12-13-53_10000.npy',y='zero') #prior 
            ba('place_Widow250PlaceTray-v0_5K_save_all_noise_0.1_2021-04-03T12-14-02_4750.npy') #task
        elif args.buffer == 8:
            path = '/nfs/kun1/users/asap7772/cog_data/'
            ba('pickplace_prior.npy',y='zero') #prior 
            path = p_data_path
            ba('place_Widow250PlaceTray-v0_5K_save_all_noise_0.1_2021-04-03T12-14-02_4750.npy') #task
        elif args.buffer == 9:
            path = p_data_path
            ba('pick_Widow250PickTray-v0_10K_save_all_noise_0.1_2021-04-03T12-13-53_10000.npy',y='zero') #prior 
            path = '/nfs/kun1/users/asap7772/cog_data/'
            ba('pickplace_task.npy') #task
        elif args.buffer == 10:
            path = '/nfs/kun1/users/asap7772/cog_data/'
            ba('pickplace_prior.npy',y='zero')
            ba('pickplace_task.npy') #task
        elif args.buffer == 11:
            path  = p_data_path
            ba('coglike_prior_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-00_10000.npy', y='zero')
            ba('coglike_task_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-05_10000.npy')
        elif args.buffer == 12:
            path  = p_data_path
            ba('coglike_prior_linking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-05T11-11-02_9250.npy', y='zero')
            ba('coglike_task_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-05_10000.npy')
        elif args.buffer == 13:
            path  = p_data_path
            ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy', y='zero',p=args.prob)
            ba('coglike_task_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-03T17-32-05_10000.npy',p=args.prob)
        elif args.buffer == 14:
            path  = p_data_path
            ba('prior_reset5_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-08_10000.npy', y='zero')
            ba('task_reset5_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-17_9000.npy')
        elif args.buffer == 15:
            path  = p_data_path
            ba('prior_reset10_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-23_10000.npy', y='zero')
            ba('task_reset10_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-28_10000.npy')
        elif args.buffer == 16:
            path  = p_data_path
            ba('prior_reset100_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-35_10000.npy', y='zero')
            ba('task_reset100_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T13-48-43_10000.npy')
        elif args.buffer == 17:
            path  = p_data_path
            ba('prior_reset2_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-56-50_8000.npy',y='zero')
            ba('task_reset2_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-56-55_10000.npy')
        elif args.buffer == 18:
            path  = p_data_path
            ba('prior_reset3_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-01_10000.npy',y='zero')
            ba('task_reset3_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-10_10000.npy')
        elif args.buffer == 19:
            path  = p_data_path
            ba('prior_reset1000_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-17_9000.npy',y='zero')
            ba('task_reset1000_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-38_10000.npy')
        elif args.buffer == 20:
            path  = p_data_path
            ba('prior_reset10000_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-44_10000.npy',y='zero')
            ba('task_reset10000_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-52_9000.npy')
        elif args.buffer == 21:
            path  = p_data_path
            ba('prior_resetinf_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-57-59_9000.npy',y='zero')
            ba('task_resetinf_Widow250DoubleDrawerGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-08T10-58-08_10000.npy')
        elif args.buffer == 22:
            ba('closed_drawer_prior.npy',p=args.prob,y='zero')
            ba('drawer_task.npy',p=args.prob)
        elif args.buffer == 23:
            path  = p_data_path
            ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy', y='zero')
            ba('randobj_2_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-04-15T14-05-01_10000.npy')
        elif args.buffer == 24:
            path  = p_data_path
            ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy', y='zero')
            ba('randobj_5_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-04-15T14-05-10_10000.npy')
        elif args.buffer == 25:
            path  = p_data_path
            ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy', y='zero')
            ba('randobj_10_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-04-15T14-05-18_9000.npy')
        elif args.buffer == 26:
            path  = p_data_path
            ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy',p=args.prob, y='zero')
            ba('coglike_task_noise0.1_Widow250DoubleDrawerGraspNeutral-v0_5K_save_all_noise_0.1_2021-04-23T02-22-30_4750.npy',p=args.prob,)
        elif args.buffer == 27:
            path  = p_data_path
            ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy',p=args.prob, y='zero')
            ba('coglike_task_noise0.15_Widow250DoubleDrawerGraspNeutral-v0_5K_save_all_noise_0.15_2021-04-23T02-22-39_4625.npy',p=args.prob,)
        elif args.buffer == 28:
            path  = p_data_path
            ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy',p=args.prob, y='zero')
            ba('coglike_task_noise0.2_Widow250DoubleDrawerGraspNeutral-v0_5K_save_all_noise_0.2_2021-04-23T02-22-44_4875.npy',p=args.prob,)
        elif args.buffer == 28:
            ba('coglike_prior_manuallinking_Widow250DoubleDrawerOpenGraspNeutral-v0_10K_save_all_noise_0.1_2021-04-06T00-36-15_10000.npy',p=args.prob, y='zero')
            ba('coglike_task_noise0.2_Widow250DoubleDrawerGraspNeutral-v0_5K_save_all_noise_0.2_2021-04-23T02-22-44_4875.npy',p=args.prob,)
        elif args.buffer == 29:
            ba('pickplace_prior.npy', p=args.prob,y='zero')
            ba('pickplace_task.npy', p=args.prob)
        elif args.buffer == 30:
            path  = p_data_path
            ba('pick_10obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-26_4500.npy', p=args.prob,y='zero')
            ba('place_10obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-31_4875.npy', p=args.prob)
        elif args.buffer == 31:
            path  = p_data_path
            ba('pick_5obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-36_4750.npy', p=args.prob,y='zero')
            ba('place_5obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-39_4750.npy', p=args.prob)
        elif args.buffer == 32:
            path  = p_data_path
            ba('pick_2obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-43_5000.npy', p=args.prob,y='zero')
            ba('place_2obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-16-49_5000.npy', p=args.prob)
        elif args.buffer == 33:
            ba('blocked_drawer_1_prior.npy', p=args.prob,y='zero')
            ba('drawer_task.npy', p=args.prob)
        elif args.buffer == 34:
            path = ''
            if args.azure:
                ba(os.path.join(os.expand_user('~'), 'grasping35obj', 'may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48_20000.npy'))
                ba(os.path.join(os.expand_user('~'), 'grasping35obj', 'may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48_20000.npy'))
            else:
                ba('/nfs/kun1/users/avi/scripted_sim_datasets/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48_20000.npy')
                ba('/nfs/kun1/users/avi/scripted_sim_datasets/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48/may11_Widow250OneObjectGraspTrain-v0_20K_save_all_noise_0.1_2021-05-11T16-56-48_20000.npy')
        elif args.buffer == 35:    
            path  = p_data_path
            ba('pick_35obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-05-07T01-17-10_4375.npy', p=args.prob, y='zero')
            ba('place_35obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-17-42_4875.npy', p=args.prob)
        elif args.buffer == 36:
            path  = p_data_path
            ba('pick_20obj_Widow250PickTrayMult-v0_5K_save_all_noise_0.1_2021-05-07T01-17-01_4625.npy', p=args.prob,
               y='zero')
            ba('place_20obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-06-14T21-53-31_5000.npy', p=args.prob)
        elif args.buffer == 37:
            path  = p_data_path
            ba('drawer_prior_multobj_Widow250DoubleDrawerOpenGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-06-23T11-52-07_10000.npy', p=args.prob, y='zero')
            ba('drawer_task_multobj_Widow250DoubleDrawerGraspNeutralRandObj-v0_10K_save_all_noise_0.1_2021-06-23T11-52-15_9750.npy', p=args.prob)
        elif args.buffer == 9000:
            variant['debug'] = True
            path  = p_data_path
            ba('debug.npy',y='noise')
            ba('debug.npy',y='noise')
        elif args.buffer == 9001: #for testing wandb code
            variant['debug'] = False 
            path  = p_data_path
            ba('debug.npy',y='noise')
            ba('debug.npy',y='noise')

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
