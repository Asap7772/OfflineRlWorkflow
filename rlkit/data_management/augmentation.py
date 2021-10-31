from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize
import torchvision.transforms.functional as F
import rlkit.torch.pytorch_util as ptu
import sys
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms

def batch_crop(x, img_size = (3,64,64)):
    flag = type(x) != torch.Tensor
    if flag:
        x = torch.from_numpy(x)
    x = x.detach().clone()
    y = x.reshape(-1, *img_size).float()
    for i in range(y.shape[0]):
        x[i] = crop(y[i])
    if flag:
        return x.cpu().numpy()
    else:
        return x

def batch_warp(x, mat, img_size = (3,64,64)):
    flag = type(x) != torch.Tensor
    if flag:
        x = torch.from_numpy(x)
    
    x = x.detach().clone()
    y = x.reshape(-1, *img_size).float()

    for i in range(y.shape[0]):
        x[i] = warp(y[i], mat)

    if flag:
        return x.cpu().numpy()
    else:
        return x

def warp(x, mat, SIZE = 64):
    def to_numpy(obs_img):
        if type(obs_img) == torch.Tensor:
            from torchvision import transforms
            im_new = transforms.ToPILImage()(obs_img.float().cpu())
        else:
            im_new = obs_img
        return np.array(im_new)

    im = to_numpy(x)
    warped = cv2.warpPerspective(im, mat, (im.shape[1], im.shape[0]))

    def revert(img):
        from PIL import Image
        img = Image.fromarray(img)
        img = np.array(img)
        img = img*1.0/255
        img = img.transpose([2,0,1]) #.flatten()
        return torch.from_numpy(img).float()

    return revert(warped).flatten()

def crop(x, SIZE = 64):
    jitter = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
    cropper = RandomResizedCrop((SIZE, SIZE), (0.9, 1.0), (0.9, 1.1))
    c = cropper.get_params(x, (0.9, 1.0), (0.9, 1.1))
    j = jitter.get_params((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
    
    x1 = transforms.ToPILImage()(x)
    # x1 = F.resized_crop(x, c[0], c[1], c[2], c[3], (SIZE, SIZE), Image.ANTIALIAS)
    x1 = np.array(j(x1)) / 255
    img1 = torch.from_numpy(x1.transpose([2, 0, 1]).flatten())
    return img1

if __name__ == '__main__':
    from rlkit.data_management.load_buffer_real import *

    import matplotlib.pyplot as plt
    def plot_img(obs_img, save='a.png'):
        plt.figure()
        if type(obs_img) == torch.Tensor:
            im_new = transforms.ToPILImage()(obs_img)
        else:
            im_new = obs_img
        plt.imshow(im_new)
        plt.savefig('/nfs/kun1/users/asap7772/cog/' + save)
        plt.show()

    observation_key = 'image'
    paths = []
    data_path = '/nfs/kun1/users/ashvin/data/val_data'

    replay_buffer = get_buffer(observation_key=observation_key)
    paths.append((os.path.join(data_path,'fixed_pot_demos.npy'), os.path.join(data_path,'fixed_pot_demos_putlidon_rew.pkl')))
    for path, rew_path in paths:
        load_path(path, rew_path, replay_buffer)
    batch = replay_buffer.random_batch(256)
    
    # import ipdb; ipdb.set_trace()
    images = torch.from_numpy(batch['observations']).float()
    plot_img(images[0].reshape(3,64,64), save='a.png')
    while True:
        import ipdb; ipdb.set_trace()
        images_new = batch_crop(images).reshape(-1,3,64,64)
        plot_img(images_new[0], save='b.png')
