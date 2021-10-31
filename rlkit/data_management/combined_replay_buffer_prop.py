import numpy as np
import random
import ipdb
# import kornia
import torch

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
import pickle
# from rlkit.data_management.load_buffer import get_buffer_size, add_data_to_buffer

import roboverse
buff1 = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-19T23-15-41.npy'
buff2 = '/nfs/kun1/users/avi/imitation_datasets/scripted_Widow250MultiObjectGraspTrain-v0_2020-12-26T22-49-57.npy'


def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions

def add_data_to_buffer(data, replay_buffer):

    for j in range(len(data)):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_images(data[j]['observations']),
            next_observations=process_images(
                data[j]['next_observations']),
        )
        replay_buffer.add_path(path)

def process_images(observations):
    output = []
    for i in range(len(observations)):
        image = observations[i]['image']
        if len(image.shape) == 3:
            image = np.transpose(image, [2, 0, 1])
            image = (image.flatten())/255.0
        else:
            print('image shape: {}'.format(image.shape))
            raise ValueError
        output.append(dict(image=image))
    return output

class CombinedReplayBuffer2(ReplayBuffer):
    def __init__(self, buffers, p = 0.3, widowx=False, kuka=False, data_aug=False, state_dim=None, img_dim=(3,48,48), online=False, online_only = False, online_env=None, online_obs_keys = None, state=False, unaltered=False):
        super().__init__()
        self.buffers = buffers
        self.data_aug = data_aug
        self.img_dim = img_dim
        self.state_dim = state_dim
        self.unaltered = unaltered

        if type(p) is float:
            p = [p] * len(buffers)
        else:
            p = [max(min(x,1), 0) for x in p] #ensure valid

        assert len(buffers) == len(p)
        
        self.p = np.array(p)
        self.sizes = np.round(np.array([buff._size for buff in self.buffers]) * p).astype(int)
        for i in range(len(self.buffers)):
            self.buffers[i]._actions = self.buffers[i]._actions[:self.sizes[i]]
            self.buffers[i]._rewards = self.buffers[i]._rewards[:self.sizes[i]] 
            self.buffers[i]._terminals = self.buffers[i]._terminals[:self.sizes[i]] 
            for key in self.buffers[i]._obs:
                self.buffers[i]._obs[key] = self.buffers[i]._obs[key][:self.sizes[i]]
                self.buffers[i]._next_obs[key] = self.buffers[i]._next_obs[key][:self.sizes[i]] 
            self.buffers[i]._size = self.sizes[i]
        
        self.online = online
        self.online_started=False
        self.online_only = online_only

        if self.online:
            self.online_pool = ObsDictReplayBuffer(2000*30, online_env, observation_keys=online_obs_keys)
        self.widowx = widowx
        self.kuka = kuka
        self._size = sum(self.sizes)
        self.sizes = self.sizes.astype(float)/self._size #normalize for proportion
        self.state = state

    def add_sample(self, observation, action, reward, next_observation, terminal, **kwargs):
        pass

    def terminate_episode(self):
        for x in self.buffers:
            ep_term = x.terminate_episode()
        return ep_term

    def num_steps_can_sample(self, **kwargs):
        return sum([x.num_steps_can_sample() for x in self.buffers])

    def add_path(self, path):
        if self.online:
            self.online_started = True
            self.online_pool.add_path(path)

    def random_batch(self, batch_size, *args, **kwargs):
        if self.online_only and self.online_started:
            if hasattr(self, 'buffers'):
                delattr(self, 'buffers')
            return self.online_pool.random_batch(batch_size)

        if self.online and self.online_started:
            p_temp = np.array(self.p.tolist() + [1])
            buffers_temp = list(self.buffers) + [self.online_pool]
            sizes_temp = np.round(np.array([buff._size for buff in buffers_temp]) * p_temp)
            sizes_temp = sizes_temp/sum(sizes_temp)

            batch_dist = np.round(batch_size * sizes_temp)
            batch_dist = np.maximum(batch_dist, np.ones_like(batch_dist))
            max_dim=batch_dist.argmax()
            batch_dist[max_dim] = batch_size - sum(batch_dist) + batch_dist[max_dim]  #ensure adds to batch size
            batch_dist = batch_dist.astype(int)
            batches = []
            
            for i in range(len(buffers_temp)):
                curr = buffers_temp[i].random_batch(batch_dist[i], *args, **kwargs)
                batches.append(curr)
        else:
            batch_dist = np.round(batch_size * self.sizes)
            batch_dist = np.maximum(batch_dist, np.ones_like(batch_dist))
            max_dim=batch_dist.argmax()
            batch_dist[max_dim] = batch_size - sum(batch_dist) + batch_dist[max_dim] #ensure adds to batch size
            batch_dist = batch_dist.astype(int)

            batches = []
            for i in range(len(self.buffers)):
                curr = self.buffers[i].random_batch(batch_dist[i], *args, **kwargs)
                batches.append(curr)

        if self.online and (self.widowx or self.kuka):
            if self.widowx:
                batches[-1]['actions'] = batches[-1]['actions'][:,:-1]
            batches[-1]['actions'] = np.concatenate((batches[-1]['actions'][:, :3], batches[-1]['actions'][:, -1:]), axis = 1)

        # import ipdb; ipdb.set_trace()
        batch = self.merge_batch(batches)

        if self.unaltered:
            if self.widowx:
                batch['actions'] = batch['actions'][:, :5]
        else:
            if self.widowx:
                batch['actions'] = np.concatenate((batch['actions'][:,:3], np.zeros_like(batch['actions'][:,3:4]), batch['actions'][:,3:4], np.zeros_like(batch['actions'][:,3:4])), axis =1)
            elif self.kuka:
                batch['actions'] = np.concatenate((batch['actions'][:,:3], np.zeros_like(batch['actions'][:,3:4]), batch['actions'][:,3:4]), axis =1)
            else:
                batch['actions'] = batch['actions']
            if self.data_aug:
                batch['observations'] = self.color_jitter(batch['observations'])
                batch['next_observations'] = self.color_jitter(batch['next_observations'])

        batch['batch_dist'] = batch_dist

        return batch

    def color_jitter(self, imgs):
        tens = torch.from_numpy(imgs)
        tens = tens.reshape((-1,) + self.img_dim)
        tens = kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5)(tens)
        return tens.reshape((-1, np.prod(self.img_dim))).numpy()

    def merge_batch(self, batches):
        # min_state = min(batches, key = lambda x: x['observations_state'].shape[-1])['observations_state'].shape[-1]
        min_obs = min(batches, key = lambda x: x['observations'].shape[-1])['observations'].shape[-1]
        min_action = min(batches, key = lambda x: x['actions'].shape[-1])['actions'].shape[-1]
        batch = {}
        for b1 in batches:
            for key in b1:
                if key in batch:
                    if key == 'observations' or key == 'next_observations': batch[key] = np.concatenate((batch[key][:,:min_obs], b1[key][:,:min_obs])) # in the image case don't care about obs size
                    # elif key == 'observations_state' or key == 'next_observations_state': batch[key] = np.concatenate((batch[key][:,:min_state], b1[key][:,:min_state])) # in the image case don't care about obs size
                    elif key == 'actions': batch[key] = np.concatenate((batch[key][:,:min_action], b1[key][:,:min_action])) # in the image case don't care about obs size
                    else: batch[key] = np.concatenate((batch[key], b1[key]))
                else: batch[key] = b1[key]

        if self.state:
            batch['observations'], batch['observations_state'] = batch['observations_state'], batch['observations']
            batch['next_observations'], batch['next_observations_state'] = batch['next_observations_state'], batch['next_observations']
            if self.state_dim and batch['observations'].shape[-1] < self.state_dim:
                d = batch['observations'].copy()
                batch['observations'] = np.zeros((batch['observations'].shape[0], self.state_dim))
                batch['observations'][:,:d.shape[-1]] = d

                d = batch['next_observations'].copy()
                batch['next_observations'] = np.zeros((batch['next_observations'].shape[0], self.state_dim))
                batch['next_observations'][:,:d.shape[-1]] = d

        return batch

    def get_diagnostics(self):
        dct = {}
        for buffer in self.buffers:
            dct = self.merge_dict(dct, buffer.get_diagnostics())
        return dct

    def get_snapshot(self):
        dct = {}
        for buffer in self.buffers:
            dct = self.merge_dict(dct, buffer.get_snapshot())
        return dct

    def merge_dict(self, dict1, dict2): 
        return {**dict1, **dict2}  

    def end_epoch(self, epoch):
        ret_val = None
        for buffer in self.buffers:
            ret_val = buffer.end_epoch(epoch)
        return ret_val

    def switch_online(self):
        ret_val = None
        for buffer in self.buffers:
            ret_val = buffer.switch_online(epoch)
        return ret_val

if __name__ == "__main__":
    extra_buffer_size = 100
    buff_locs = [buff1, buff2]
    
    expl_env = roboverse.make('Widow250MultiObjectGraspTrain-v0', transpose_image=True)
    buffers = []
    for buff_loc in buff_locs:
        with open(buff_loc, 'rb') as f:
            data = np.load(f, allow_pickle=True)

        num_transitions = get_buffer_size(data)
        buffer_size = num_transitions + extra_buffer_size
        
        observation_key = 'image'

        replay_buffer = ObsDictReplayBuffer(
            buffer_size,
            expl_env,
            observation_key=observation_key,
            color_segment=True,
            target_segment='fixed_other'
        )
        add_data_to_buffer(data, replay_buffer)
        print('Data loaded from npy file', replay_buffer._top)
        buffers.append(replay_buffer)

    buff_new =  CombinedReplayBuffer2(*buffers, p=[1,0.2])
    import ipdb; ipdb.set_trace()
    buff_new.random_batch(5)