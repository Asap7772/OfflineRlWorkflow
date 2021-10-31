import matplotlib.pyplot as plt
import numpy as np
import os
from rlkit.data_management.load_buffer_real import *
import pickle
from railrl.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

def plot_traj(imgs, rows, cols, rewards = None):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.set_figheight((6 if rewards is not None else 5)*rows)
    fig.set_figwidth(5*cols)
    if rewards is None:
        for img, ax in zip(imgs, axes.ravel()):
            ax.imshow(img)
            ax.axis('off')
    else:
        for img, ax, rew in zip(imgs, axes.ravel(), rewards):
            ax.imshow(img)
            ax.title.set_text('Reward = ' + str(rew))
            ax.axis('off')
    fig.tight_layout()
    return fig

from os.path import expanduser
path = '/nfs/kun1/users/stephentian/on_policy_longer_1_26_buffers/move_tool_obj_together_fixed_6_2_train.pkl'

print('loading')
replay_buffer = pickle.load(open(path,'rb'))
print('done loading')

replay_buffer_new = ObsDictReplayBuffer(replay_buffer.max_size, replay_buffer.env, dummy=True)
replay_buffer_new.load_from(replay_buffer)

replay_buffer = replay_buffer_new

replay_buffer.color_jitter=True
replay_buffer.warp_img=True

# p = '/nfs/kun1/users/asap7772/prior_data/'
# buff_name = 'place_35obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-17-42_4875.npy'
save_path = '/nfs/kun1/users/asap7772/cog/film_strip_out'
num_traj = 50
timesteps_per_traj = 10
skip_by = 1
buff_name = 'move_tool_obj_together_fixed_6_2_train.pkl'

def process_image(img):
    image_dataset = torch.from_numpy(img.reshape(3,64,64)).float()
    if type(image_dataset) == torch.Tensor:
        from torchvision import transforms
        im_new = transforms.ToPILImage()(image_dataset.detach().cpu())
    else:
        im_new = image_dataset
    return im_new

data = replay_buffer._obs['image']
rew = replay_buffer._rewards

for i in range(num_traj):
    imgs = [data[i*num_traj + j] for j in range(timesteps_per_traj)] 
    imgs = [process_image(x) for x in imgs[::skip_by]]
    rewards = [rew[i*num_traj + j] for j in range(timesteps_per_traj)][::skip_by]
    fig = plot_traj(imgs, 1, len(imgs), rewards = rewards)
    if not os.path.exists(os.path.join(save_path, buff_name.split('.')[0])):
        os.mkdir(os.path.join(save_path, buff_name.split('.')[0]))
    path_out = os.path.join(save_path, buff_name.split('.')[0], str(i)+'.png')
    print(path_out)
    plt.savefig(path_out)
    plt.close()
