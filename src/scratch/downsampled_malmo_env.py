import logging

import numpy as np

from scratch.image import resize, rgb2gray
import gym
from malmoenv.core import Env


class DownsampledMalmoEnv(Env):
    """
    Generate RGB frame observation resizing to the specified width/height and depth
    """

    def __init__(self, height, width, grayscale, *args, normalize=True, add_health=False, **kwargs):
        super().__init__(*args, **kwargs)

        assert height > 0, 'height should be > 0'
        assert width > 0, 'width should be > 0'

        self._height = height
        self._width = width
        self._gray = bool(grayscale)
        self._normalize = normalize
        self._add_health = add_health

        if self._gray:
            if self._normalize:
                self._space = gym.spaces.Box(0.0, 1.0,
                                             (self._height, self._width, 1), np.float32)
            else:
                self._space = gym.spaces.Box(0, 255,
                                             (self._height, self._width, 1), np.uint8)
        else:
            if self._normalize:
                self._space = gym.spaces.Box(0.0, 1.0,
                                             (self._height, self._width, 3), np.float32)
            else:
                self._space = gym.spaces.Box(0, 255,
                                             (self._height, self._width, 3), np.uint8)

        self.observation_space = self._space

    def reset(self):
        """
        Reset the environment.
        """
        raw_obs = super().reset()
        return self.convert_observation(raw_obs)

    def step(self, action):
        obs, rew, done, info = super().step(action)
        obs = self.convert_observation(obs)
        return obs, rew, done, {}

    def convert_observation(self, img):
        if len(img) != 0:
            img = resize(img, self._space.shape[:2])
            if self._gray:
                # rgb2gray may collapse the last dimension - recover it
                if len(img.shape) < 3:
                    img = img.reshape(img.shape + (1,))
            if self._normalize:
                return (img / 255.0).astype(np.float32)
            return img.astype(self._space.dtype)

        logging.info("none")
        return np.zeros(self._space.shape, self._space.dtype)
