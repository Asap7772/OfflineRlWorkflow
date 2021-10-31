import os
import torch
import argparse
from rlkit.torch.conv_networks import CNN, ConcatCNN, ConcatBottleneckCNN, TwoHeadCNN
from rlkit.data_management.load_buffer_real import *
import rlkit.torch.pytorch_util as ptu

ptu.set_gpu_mode(True)
def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return
enable_gpus("7")

parser = argparse.ArgumentParser(description='Process data')
parser.add_argument('--path', type=str)
parser.add_argument('--buffer', type=int, default=0)
parser.add_argument('--bottleneck', action='store_true')
parser.add_argument('--bottleneck', action='store_true')
parser.add_argument('--azure', action='store_true')
args = parser.parse_args()

path = args.path
dir_inner = os.listdir(path)[0]
new_path = os.path.join(path, dir_inner, 'model_pkl')

lst = sorted(os.listdir(new_path), key = lambda x: int(x.split('.')[0]))
action_dim = 4

observation_key = 'image'
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
else:
    assert False

if args.buffer in [4]:
    replay_buffer = pickle.load(open(path,'rb'))
else:
    replay_buffer = get_buffer(observation_key=observation_key)
    for path, rew_path in paths:
        load_path(path, rew_path, replay_buffer)

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
)
cnn_params.update(
    input_width=64,
    input_height=64,
    input_channels=3,
    output_size=1,
    added_fc_input_size=action_dim,
)

epochs = []
dot_prods = []
cos_sims = []

for p in lst:
    epoch = int(p.split('.')[0])
    print('done', epoch)

    if args.bottleneck:
        qf1 = ConcatBottleneckCNN(action_dim, bottleneck_dim=16,deterministic=False, width=64, height=64).to(ptu.device)
        qf1.output_conv_channels = True
    else:
        qf1 = ConcatCNN(**cnn_params).to(ptu.device)
        qf1.output_conv_channels = True
    
    parameters = torch.load(os.path.join(new_path, p))
    qf1.load_state_dict(parameters['qf1_state_dict'])

    batch = replay_buffer.random_batch(256)

    s, a = torch.from_numpy(batch['observations']).float().to(ptu.device), torch.from_numpy(batch['actions']).float().to(ptu.device)
    sp, ap = torch.from_numpy(batch['next_observations']).float().to(ptu.device), torch.from_numpy(batch['next_actions']).float().to(ptu.device)

    feat_curr = qf1(s,a).flatten(start_dim=1)
    feat_next = qf1(sp, ap).flatten(start_dim=1)
    
    dot_prod = (feat_curr*feat_next).sum(axis=1)
    cos_sim = torch.nn.CosineSimilarity()(feat_curr, feat_next)
    
    dot_prods.append(dot_prod.mean().item())
    cos_sims.append(cos_sim.mean().item())
    epochs.append(epoch)
    
    del qf1
    del s, a, sp, ap
    del feat_curr, feat_next
    del dot_prod, cos_sim
    torch.cuda.empty_cache()

import ipdb; ipdb.set_trace()



