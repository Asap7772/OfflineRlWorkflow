import matplotlib.pyplot as plt
import numpy as np
import os

def plot_traj(imgs, rows, cols):
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    fig.set_figheight(5*rows)
    fig.set_figwidth(5*cols)
    for img, ax in zip(imgs, axes.ravel()):
        ax.imshow(img)
        ax.axis('off')
    fig.tight_layout()
    return fig

# p = '/nfs/kun1/users/asap7772/prior_data/'
p = '/nfs/kun1/users/asap7772/cog_data/'
# buff_name = 'place_35obj_Widow250PlaceTrayMult-v0_5K_save_all_noise_0.1_2021-04-30T01-17-42_4875.npy'
buff_name = 'drawer_task.npy'
p += buff_name
save_path = '/nfs/kun1/users/asap7772/cog/film_strip_out'
num_traj = 10
skip_by = 3

with open(p, 'rb') as f:
    data = np.load(f, allow_pickle=True)

for i in range(num_traj):
    imgs = [x['image'] for x in data[i]['observations']] 
    imgs = imgs[::skip_by]
    fig = plot_traj(imgs, 1, len(imgs))
    if not os.path.exists(os.path.join(save_path, buff_name.split('.')[0])):
        os.mkdir(os.path.join(save_path, buff_name.split('.')[0]))
    path_out = os.path.join(save_path, buff_name.split('.')[0], str(i)+'.png')
    print(path_out)
    plt.savefig(path_out)
