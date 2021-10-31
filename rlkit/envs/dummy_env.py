from gym import Env
import gym.spaces
import numpy as np

class DummyEnv(Env):
    def __init__(self, image_shape = (64, 64, 3), state_shape=(3,)) -> None:
        super().__init__()

        self.image_shape = image_shape
        self.state_shape = state_shape
        self.action_shape = (4,)

        self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(0, 255, self.image_shape),
                'state': gym.spaces.Box(-1, 1, self.state_shape)
            })

        self.action_space = gym.spaces.Box(-1, 1, self.action_shape)

    def step(self, action):
        return {
            'image': np.zeros(self.image_shape).flatten(),
            'state': np.zeros(self.state_shape)
        }, 0, 0, {}

    def reset(self):
        return {
            'image': np.zeros(self.image_shape).flatten(),
            'state': np.zeros(self.state_shape)
        }

