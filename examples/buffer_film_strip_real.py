import matplotlib.pyplot as plt
import numpy as np
import os
from rlkit.data_management.load_buffer_real import *

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

buffer_num = 4
observation_key = 'image'
paths = []
data_path = '/nfs/kun1/users/asap7772/real_data_drawer/val_data_relabeled/'
if buffer_num == 0:
    print('lid on')
    buff_name = 'fixed_pot_on_demos_latent.npy'
    paths.append((os.path.join(data_path,'fixed_pot_demos_latent.npy'), os.path.join(data_path,'fixed_pot_demos_lidon_rew_handlabel_06_13.pkl')))
elif buffer_num == 1:
    print('lid off')
    buff_name = 'fixed_pot_off_demos_latent.npy'
    paths.append((os.path.join(data_path,'fixed_pot_demos_latent.npy'), os.path.join(data_path,'fixed_pot_demos_lidoff_rew_handlabel_06_13.pkl')))
elif buffer_num == 2:
    print('tray')
    buff_name = 'fixed_tray_demos_latent.npy'
    paths.append((os.path.join(data_path,'fixed_tray_demos_latent.npy'), os.path.join(data_path,'fixed_tray_demos_rew.pkl')))
elif buffer_num == 3:
    print('drawer')
    buff_name = 'fixed_drawer_demos_latent.npy'
    paths.append((os.path.join(data_path,'fixed_drawer_demos_latent.npy'), os.path.join(data_path,'fixed_drawer_demos_draweropen_rew_handlabel_06_13.pkl')))
elif buffer_num == 4:
    print('Albert Pick Place')
    buff_name = 'real_pick_place.npy'
    data_path = '/nfs/kun1/users/albert/realrobot_datasets/combined_2021-06-03_21_36_48_labeled.pkl'
    paths.append((data_path, None))
else:
    assert False

for path, rew_path in paths:
    data = load_data(path, rew_path, bc=True)


# p = '/nfs/kun1/users/asap7772/prior_data/'
# buff_name = 'place_35obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-17-42_4875.npy'
save_path = '/nfs/kun1/users/asap7772/cog/film_strip_out'
num_traj = 10
skip_by = 2 if buffer_num == 4 else 10 

def process_image(img):
    image_dataset = torch.from_numpy(img.reshape(3,64,64)).float()
    if type(image_dataset) == torch.Tensor:
        from torchvision import transforms
        im_new = transforms.ToPILImage()(image_dataset.detach().cpu())
    else:
        im_new = image_dataset
    return im_new if buffer_num == 4 else im_new.rotate(270)

for i in range(num_traj):
    imgs = [x['image' if buffer_num == 4 else 'image_observation'] for x in data[i]['observations']] 
    imgs = [process_image(x) for x in imgs[::skip_by]]
    rewards = data[i]['rewards'][::skip_by]
    fig = plot_traj(imgs, 1, len(imgs), rewards = rewards)
    if not os.path.exists(os.path.join(save_path, buff_name.split('.')[0])):
        os.mkdir(os.path.join(save_path, buff_name.split('.')[0]))
    path_out = os.path.join(save_path, buff_name.split('.')[0], str(i)+'.png')
    print(path_out)
    plt.savefig(path_out)
