
import os
from pickle import FALSE
import numpy as np
from rlkit.data_management.load_buffer import get_buffer_size, add_data_to_buffer
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
import roboverse
import torch
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN, VQVAEEncoderConcatCNN, \
    ConcatBottleneckVQVAECNN, VQVAEEncoderCNN
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.core import np_to_pytorch_batch
import rlkit.torch.pytorch_util as ptu
ptu.set_gpu_mode(True)


import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import gc


def format_func(value, tick_number):
    return(str(int(value // 100) * 0.1)[:3] + 'M')

def format_func_y(value, tick_number):
    return (str(int(value))[:4] + 'K')

def configure_matplotlib(matplotlib):
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath}',
    })

    matplotlib.rc('font', family='serif', serif='cm10')
    # matplotlib.rcParams['font.weight']= 'heavy
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size': 15}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['lines.linewidth'] = 2.5

configure_matplotlib(matplotlib)

def load_buffer(buffer=0):
    home = os.path.expanduser("~")
    p_data_path =  os.path.join(home, 'prior_data/')
            
    path = '/nfs/kun1/users/asap7772/cog_data/'
    buffers = []
    ba = lambda x, p=0, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))

    if buffer == 0:
        ba('blocked_drawer_1_prior.npy', p=1,y='zero')
        ba('drawer_task.npy', p=1)
    else:
        ba('pickplace_prior.npy',y='zero')
        ba('pickplace_task.npy') 

    print(buffers)

    eval_env = roboverse.make('Widow250DoubleDrawerCloseOpenGraspNeutral-v0', transpose_image=True)

    variant = dict()
    variant['prior_buffer'] = buffers[0][0]

    with open(variant['prior_buffer'], 'rb') as f:
        data_prior = np.load(f, allow_pickle=True)
    buffer_size = get_buffer_size(data_prior)

    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        eval_env,
        observation_key='image',
    )

    add_data_to_buffer(data_prior, replay_buffer, initial_sd=True)
    return replay_buffer, eval_env

def load_buffer_full(buffer=0):
    home = os.path.expanduser("~")
    p_data_path =  os.path.join(home, 'prior_data/')
            
    path = '/nfs/kun1/users/asap7772/cog_data/'
    buffers = []
    ba = lambda x, p=0, y=None: buffers.append((path+x,dict(p=p,alter_type=y,)))
    
    if buffer == 0:
        ba('blocked_drawer_1_prior.npy', p=1,y='zero')
        ba('drawer_task.npy', p=1)
    else:
        ba('pickplace_prior.npy',y='zero')
        ba('pickplace_task.npy')

    print(buffers)

    eval_env = roboverse.make('Widow250DoubleDrawerCloseOpenGraspNeutral-v0', transpose_image=True)

    variant = dict()
    variant['prior_buffer'] = buffers[0][0]

    with open(variant['prior_buffer'], 'rb') as f:
        data_prior = np.load(f, allow_pickle=True)
    buffer_size = get_buffer_size(data_prior)


    variant['task_buffer'] = buffers[1][0]

    with open(variant['task_buffer'], 'rb') as f:
        data_task = np.load(f, allow_pickle=True)
    buffer_size =  get_buffer_size(data_prior) + 2*get_buffer_size(data_task)

    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        eval_env,
        observation_key='image',
    )

    add_data_to_buffer(data_prior, replay_buffer, initial_sd=False)
    add_data_to_buffer(data_task, replay_buffer, initial_sd=False)
    return replay_buffer, eval_env

def load_qfunc_pol(eval_env, path):
    action_dim = eval_env.action_space.low.size
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
    )
    cnn_params.update(
        input_width=48,
        input_height=48,
        input_channels=3,
        output_size=1,
        added_fc_input_size=action_dim,
        normalize_conv_activation=False
    )

    qfunc = ConcatCNN(**cnn_params)

    cnn_params.update(
        output_size=256,
        added_fc_input_size=0,
        hidden_sizes=[1024, 512],
    ) 
    
    policy_obs_processor = CNN(**cnn_params)
    policy = TanhGaussianPolicy(
        obs_dim=cnn_params['output_size'],
        action_dim=action_dim,
        hidden_sizes=[256, 256, 256],
        obs_processor=policy_obs_processor,
        shared_encoder=False,
    )

    parameters = torch.load(path)
    qfunc.load_state_dict(parameters['qf1_state_dict'])
    policy.load_state_dict(parameters['policy_state_dict'])
    
    qfunc.to(ptu.device)
    policy.to(ptu.device)

    return qfunc, policy

if __name__ == "__main__":
    replay_buffer, eval_env = load_buffer(buffer=1)
    replay_buffer_full, eval_env = load_buffer_full(buffer=1)

    path = '/home/asap7772/asap7772/cog_stephen/cog/data/debug-pp100-minq2/debug_pp100_minq2_2021_08_27_01_06_13_0000--s-0/model_pkl'
    # path = '/nfs/kun1/users/asap7772/cog_stephen/cog/data/debug-minq2-origcog-50traj/debug_minq2_origcog_50traj_2021_08_24_00_42_48_0000--s-0/model_pkl'

    lst = sorted([p for p in os.listdir(path) if p.endswith('.pt')], key= lambda p: int(p.split('.')[0]))
    qvals, epochs, metrics = [], [], []
    for p in lst:
        epoch = int(p.split('.')[0])
        qfunc, policy = load_qfunc_pol(eval_env, os.path.join(path, p))

        batch = np_to_pytorch_batch(replay_buffer.random_batch(256))
        batch_f = np_to_pytorch_batch(replay_buffer_full.random_batch(256))

        qval = qfunc(batch['observations'], policy(batch['observations'])[0])
        met = qfunc(batch_f['observations'], batch_f['actions'])
        
        qval = qval.detach().cpu().numpy()
        met = met.detach().cpu().numpy()

        epochs.append(epoch)
        qvals.append(qval.mean())
        metrics.append(met.mean())

        print(epoch, qval.mean(), met.mean())

        del qfunc, policy
        gc.collect()
        torch.cuda.empty_cache()  

    plt.title('Metric Comparison — Pick Place Task')
    plt.figure(figsize=(5, 5))
    fig, ax1 = plt.subplots()
    p1 = plt.plot(epochs, qvals, color='C0', label='Initial State Q Value', linewidth=2)
    p2 = plt.plot(epochs, metrics, color='C1', label='Metric 4.1', linewidth=2)

    plt.xlabel(r'Gradient steps')
    plt.ylabel(r'Metric Value')
    plt.xlim(20, 300)

    plt.tight_layout()
    plt.grid(color='grey', linestyle='dotted', linewidth=1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func_y))

    ps = [p1,p2]
    labels = ['Initial State Q Value', 'Metric 4.1']
    fig.legend(handles=[p[0] for p in ps], labels=labels,loc='upper right', bbox_to_anchor=(0.87, 0.90))

    storage_loc = '/nfs/kun1/users/asap7772/workflow_plotting/plots'
    storage_loc = os.path.join(storage_loc, 'metriccomp')
    save_path = 'metric_pp'

    if not os.path.exists(storage_loc):
        os.mkdir(storage_loc)
    f_name = save_path +'.pdf'
    f_path = os.path.join(storage_loc, f_name)
    plt.savefig(f_path)
    plt.close()

    import ipdb; ipdb.set_trace()