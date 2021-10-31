import numpy as np
from gym.spaces import Dict, Discrete

from rlkit.data_management.replay_buffer import ReplayBuffer
import typing
from rlkit.data_management.augmentation import crop, batch_crop


class ObsDictRelabelingBuffer(ReplayBuffer):
    """
    Replay buffer for environments whose observations are dictionaries, such as
        - OpenAI Gym GoalEnv environments. https://blog.openai.com/ingredients-for-robotics-research/
        - multiworld MultitaskEnv. https://githuZb.com/vitchyr/multiworld/

    Implementation details:
     - Only add_path is implemented.
     - Image observations are presumed to start with the 'image_' prefix
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
            internal_keys=None,
            goal_keys=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
    ):
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        if goal_keys is None:
            goal_keys = []
        if desired_goal_key not in goal_keys:
            goal_keys.append(desired_goal_key)
        self.goal_keys = goal_keys
        assert isinstance(env.observation_space, Dict)
        assert 0 <= fraction_goals_rollout_goals
        assert 0 <= fraction_goals_env_goals
        assert 0 <= fraction_goals_rollout_goals + fraction_goals_env_goals
        assert fraction_goals_rollout_goals + fraction_goals_env_goals <= 1
        self.max_size = max_size
        self.env = env
        self.fraction_goals_rollout_goals = fraction_goals_rollout_goals
        self.fraction_goals_env_goals = fraction_goals_env_goals
        self.ob_keys_to_save = [
            observation_key,
            desired_goal_key,
            achieved_goal_key,
        ]
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        if isinstance(self.env.action_space, Discrete):
            self._action_dim = env.action_space.n
        else:
            self._action_dim = env.action_space.low.size

        self._actions = np.zeros((max_size, self._action_dim))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_size, 1), dtype='uint8')
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        print(self.env.observation_space.spaces)
        for key in self.ob_keys_to_save + internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            type = np.float64
            if key.startswith('image'):
                type = np.uint8
            self._obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)
            self._next_obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)

        self._top = 0
        self._size = 0

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        if isinstance(self.env.action_space, Discrete):
            actions = np.eye(self._action_dim)[actions].reshape((-1, self._action_dim))
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)

        if self._top + path_len >= self.max_size:
            """
            All of this logic is to handle wrapping the pointer when the
            replay buffer gets full.
            """
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self.max_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            env_goals = self.env.sample_goals(num_env_goals)
            env_goals = preprocess_obs_dict(env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals
            resampled_goals[num_rollout_goals:last_env_goal_idx] = (
                env_goals[self.desired_goal_key]
            )
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
                new_next_obs_dict[goal_key][
                num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
        if num_future_goals > 0:
            future_obs_idxs = []
            for i in indices[-num_future_goals:]:
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice. Makes you wonder what
                # random.choice is doing
                num_options = len(possible_future_obs_idxs)
                next_obs_i = int(np.random.randint(0, num_options))
                future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
            future_obs_idxs = np.array(future_obs_idxs)
            resampled_goals[-num_future_goals:] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]

        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)
        # resampled_goals must be postprocessed as well
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        new_actions = self._actions[indices]
        """
        For example, the environments in this repo have batch-wise
        implementations of computing rewards:

        https://github.com/vitchyr/multiworld
        """

        if hasattr(self.env, 'compute_rewards'):
            new_rewards = self.env.compute_rewards(
                new_actions,
                new_next_obs_dict,
            )
        else:  # Assuming it's a (possibly wrapped) gym GoalEnv
            new_rewards = np.ones((batch_size, 1))
            for i in range(batch_size):
                new_rewards[i] = self.env.compute_reward(
                    new_next_obs_dict[self.achieved_goal_key][i],
                    new_next_obs_dict[self.desired_goal_key][i],
                    None
                )
        new_rewards = new_rewards.reshape(-1, 1)

        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]
        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
        }
        return batch

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save
        }


class ObsDictReplayBuffer(ReplayBuffer):
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.

    Implementation details:
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            ob_keys_to_save=None,
            internal_keys=None,
            observation_key='observation',
            save_data_in_snapshot=False,
            biased_sampling=False,
            bias_point=None,
            before_bias_point_probability=0.5,
            color_segment = False,
            target_segment = 'fixed_other',
            store_latent=True,
            latent_dim=720,
            color_jitter=False,
            jit_percent = 0.1,
    ):
        """

        :param max_size:
        :param env:
        :param ob_keys_to_save: List of keys to save
        """
        if ob_keys_to_save is None:
            ob_keys_to_save = []
        else:  # in case it's a tuple
            ob_keys_to_save = list(ob_keys_to_save)
        if internal_keys is None:
            internal_keys = list(observation_key) if isinstance(observation_key, tuple) else [observation_key]
        else:
            internal_keys.append(observation_key)

        self.color_jitter = color_jitter
        self.jit_percent = jit_percent
        self.internal_keys = internal_keys
        # assert isinstance(observation_keys, typing.Iterable)
        assert isinstance(env.observation_space, Dict)
        # observation_keys = tuple(observation_keys)
        self.max_size = max_size
        self.env = env
        self.ob_keys_to_save = ob_keys_to_save
        self.observation_key = observation_key
        self.save_data_in_snapshot = save_data_in_snapshot
        self.color_segment = color_segment

        # Args for biased sampling from the replay buffer
        self.biased_sampling = biased_sampling
        self.bias_point = bias_point
        self.before_bias_point_probability = before_bias_point_probability

        self._action_dim = env.action_space.low.size
        self._actions = np.zeros((max_size, self._action_dim), dtype=np.float32)
        self._next_actions = np.zeros((max_size, self._action_dim), dtype=np.float32)
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_size, 1), dtype='uint8')
        self._rewards = np.zeros((max_size, 1))
        self._mcrewards = np.zeros((max_size, 1))
        self._object_positions = np.zeros((max_size, 2))
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        print(self.env.observation_space.spaces, self.ob_keys_to_save,internal_keys)
        self.ob_spaces = self.env.observation_space.spaces

        self.target_segment = target_segment
        
        for key in self.ob_keys_to_save + internal_keys:
            if key == 'camera_orientation':
                self._obs[key] = np.zeros(
                    (max_size, 3), dtype=type)
                self._next_obs[key] = np.zeros(
                    (max_size, 3), dtype=type)
            else:    
                assert key in self.ob_spaces, \
                    "Key not found in the observation space: %s" % key
                if key.startswith('image'):
                    type = np.uint8
                self._obs[key] = np.zeros(
                    (max_size, self.ob_spaces[key].low.size), dtype=type)
                self._next_obs[key] = np.zeros(
                    (max_size, self.ob_spaces[key].low.size), dtype=type)
        
        self.store_latent = store_latent
        self.latent_dim = latent_dim
        if self.store_latent:
            self._latents = np.zeros((max_size, latent_dim), dtype=np.float32)
            self._next_latents = np.zeros((max_size, latent_dim), dtype=np.float32)
        self._top = 0
        self._size = 0

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

        if isinstance(self.env.action_space, Discrete):
            raise NotImplementedError("TODO")

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_paths_from_mdp(self, paths):
        for path in paths:
            self.add_path_from_mdp(path)

    # TODO This function is a temp hack,
    # it should be removed after changes to MdpPathCollector
    def add_path_from_mdp(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)

        # this gets rid of size 1 np arrays
        obs = [obs_[0] for obs_ in obs]
        next_obs = [next_obs_[0] for next_obs_ in next_obs]

        obs = flatten_dict(list(obs), self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(list(next_obs),
                                self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)
        self.add_processed_path(path_len, actions, terminals,
                                obs, next_obs, rewards)

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        next_actions = path["next_actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        if 'mcrewards' in path:
            mcrewards = path['mcrewards']
        else:
            mcrewards = None

        if 'object_position' in path:
            object_positions = path['object_position']

        if 'latents' in path:
            latents = path['latents']
            next_latents = path['next_latents']
        else:
            latents=next_latents=None

        path_len = len(rewards)

        actions = flatten_n(actions)
        next_actions = flatten_n(next_actions)
        
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs,
                                self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)
        self.add_processed_path(path_len, actions, terminals, obs, 
        next_obs, rewards, mcrewards=mcrewards, next_actions=next_actions,
        object_positions=object_positions if 'object_position' in path else None,
        latents=latents, next_latents=next_latents)

    def add_processed_path(self, path_len, actions, terminals,
                           obs, next_obs, rewards, mcrewards=None, 
                           next_actions=None, object_positions=None,
                           latents=None, next_latents=None):
        if self._top + path_len >= self.max_size:
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]
            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                if next_actions is not None:
                    self._next_actions[buffer_slice] = next_actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._rewards[buffer_slice] = terminals[path_slice]
                if mcrewards is not None:
                    self._mcrewards[buffer_slice] = mcrewards[path_slice]
                if object_positions is not None:
                    self._object_positions[buffer_slice] = object_positions[path_slice]
                if latents is not None:
                    self._latents[buffer_slice] = latents[path_slice]
                    self._next_latents[buffer_slice] = next_latents[path_slice]

                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][
                        path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self.max_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            self._rewards[slc] = rewards

            if next_actions is not None:
                self._next_actions[slc] = next_actions
            if mcrewards is not None:
                self._mcrewards[slc] = mcrewards
            if object_positions is not None:
                self._object_positions[slc] = object_positions
            if latents is not None:
                self._latents[slc] = latents
                self._next_latents[slc] = next_latents

            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )

        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size):
        if self.biased_sampling:
            # sample from before the "bias point" with p=before_bias_point_prob
            assert self.bias_point is not None
            indices_1 = np.random.randint(0, self.bias_point, batch_size)
            indices_2 = np.random.randint(self.bias_point, self._size, batch_size)
            biased_coin_flip = (np.random.uniform(size=batch_size) <
                                self.before_bias_point_probability) * 1
            indices = np.where(biased_coin_flip, indices_1, indices_2)
        else:
            indices = np.random.randint(0, self._size, batch_size)
        return indices

    def color_segment_img(self, im, source = [150,150,150], target = 'random', shape=(48,48,3)):
        r1, g1, b1 = 150, 150, 150 # Original value
        if target == 'random':
            r2, g2, b2 = (np.random.random((3,)) * 256).astype(int)
        elif target == 'random_control':
            r2, g2, b2 = (np.random.random((3,)) * 32).astype(int)
        elif target == 'fixed':
            r2, g2, b2  = (np.random.randint(0,2,(3,)) * 256).astype(int)
        elif target == 'fixed_other':
            r2, g2, b2  = (np.random.randint(0,2,(3,)) * 256).astype(int)
            while r2 == 256 and g2 == 256 and b2 == 256:
                r2, g2, b2  = (np.random.randint(0,2,(3,)) * 256).astype(int)
        else:
            r2, g2, b2 = target

        im = im.reshape((im.shape[0], shape[0], shape[1], shape[2]))
        red, green, blue = im[:,:,:,0], im[:,:,:,1], im[:,:,:,2]
        mask = (red > r1) & (green > g1) & (blue > b1)
        
        tmp = 255 - im[:,:, :, :3][mask]
        im[:,:, :, :3][mask] = [r2, g2, b2] + tmp

        return np.clip(im,0,255).reshape(im.shape[0], -1)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_actions = self._next_actions[indices]
        mcrewards = self._mcrewards[indices]
        object_positions=self._object_positions[indices]
        terminals = self._terminals[indices]
        if isinstance(self.observation_key, tuple):
            obs = self._obs['image'][indices]
            next_obs = self._next_obs['image'][indices]
            state = self._obs['state'][indices]
            next_state = self._next_obs['state'][indices]
        else:
            obs = self._obs[self.observation_key][indices]
            next_obs = self._next_obs[self.observation_key][indices]
            state=next_state=None


        if self.color_segment:
            self.color_segment_img(obs, target=self.target_segment)
            self.color_segment_img(next_obs, target=self.target_segment)

        if self.observation_key == 'image':
            obs = normalize_image(obs)
            next_obs = normalize_image(next_obs)
        
        if self.color_jitter and np.random.rand() < self.jit_percent:
            obs = batch_crop(obs)
            next_obs = batch_crop(next_obs)

        batch = {}

        if 'camera_orientation' in self._obs:
            data = self._obs['camera_orientation'][indices]
            batch.update({
                'camera_orientation' : data,
            })

        # import ipdb; ipdb.set_trace()
        # if 'camera_orientation' in self._obs:
        #     data = self._obs['camera_orientation'][indices]
        #     next_data = self._next_obs['camera_orientation'][indices]
        #     batch.update({
        #         'camera_orientation' : np.array(list(data.values())),
        #         'next_camera_orientation' : np.array(list(next_data.values()))
        #     })
        batch.update({
            'observations': obs,
            'actions': actions,
            'next_actions': next_actions,
            'rewards': rewards,
            'mcrewards': mcrewards,
            'object_positions': object_positions,
            'terminals': terminals,
            'next_observations': next_obs,
            'indices': np.array(indices).reshape(-1, 1),
        })

        if state is not None:
            batch.update({
                'state' : state,
                'next_state': next_state
            })
        if hasattr(self, '_latents'):
            batch.update({
                'latents': self._latents[indices],
                'next_latents': self._next_latents[indices]
            })
        return batch

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        if self.save_data_in_snapshot:
            snapshot.update({
                'observations': self.get_slice(self._obs, slice(0, self._top)),
                'next_observations': self.get_slice(
                    self._next_obs, slice(0, self._top)
                ),
                'actions': self._actions[:self._top],
                'terminals': self._terminals[:self._top],
                'rewards': self._rewards[:self._top],
                'idx_to_future_obs_idx': (
                    self._idx_to_future_obs_idx[:self._top]
                ),
            })
        return snapshot

    def get_slice(self, obs_dict, slc):
        new_dict = {}
        for key in self.ob_keys_to_save + self.internal_keys:
            new_dict[key] = obs_dict[key][slc]
        return new_dict

    def get_diagnostics(self):
        return {'top': self._top}


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }


def preprocess_obs_dict(obs_dict):
    """
    Apply internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = unnormalize_image(obs)
    return obs_dict


def postprocess_obs_dict(obs_dict):
    """
    Undo internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict


def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0


def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
