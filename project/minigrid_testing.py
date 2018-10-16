from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym

from gym_minigrid.wrappers import FullyObsWrapper
# register minigrid envs

from gym.envs.registration import register
register(
    id='MiniGrid-Empty-8x8-v1',
    entry_point='gym_minigrid.envs:EmptyEnv',
)

register(
    id='MiniGrid-KeyCorridorS6R3-v1',
    entry_point='gym_minigrid.envs:KeyCorridorS6R3',
)


class PreprocessEnv(gym.ObservationWrapper):
    """
    An environment that
    1) turns RGB images into grayscale
    2) downsamples the images
    """

    def __init__(self, env, keep_depth=True, integers=True):
        """
        Create a grayscaling wrapper.
        Args:
          env: the environment to wrap.
          keep_depth: if True, a depth dimension is kept.
            Otherwise, the output is 2-D.
          integers: if True, the pixels are in [0, 255].
            Otherwise, they are in [0.0, 1.0].
        """
        super().__init__(env)
        old_space = env.observation_space
        self._integers = integers
        self._keep_depth = keep_depth
        self.observation_space = gym.spaces.Box(self.observation(old_space.low),
                                                self.observation(old_space.high),
                                                dtype=old_space.dtype)


    def observation(self, observation):
        observation = self.to_grayscale(observation)
        # note when you downsample by 3, artifacts appear
        observation = self.downsample(observation,3)
        #observation = self.grayscale_2_onechannel(observation)
        return observation

    def grayscale_2_onechannel(self,img):
        height, width = img.shape[0], img.shape[1]
        img = np.zeros((height,width))
        nchannels = 1
        img = np.resize(img, (height, width, nchannels))
        print('RESIZED: ' ,img.shape)
        return img

    def to_grayscale(self,img):
        img = Image.fromarray(img, 'RGB').convert('L')
        img = np.array(img)
        return img


    def downsample(self,img,rate):
        return img[::rate, ::rate]






import gym
env = PreprocessEnv(FullyObsWrapper(gym.make('MiniGrid-Empty-8x8-v1')))
#env = gym.make('MiniGrid-Empty-8x8-v1')
print('FIRST obs: %s, act: %s' % (env.observation_space,env.action_space))
env.reset()
env.render()
for s in range(1000):
    if s% 100 ==0 :
        #env.render()
        pass
    if s==20:
        env.render()
        print('obs: %s, act: %s' % (env.observation_space,env.action_space))
    obs, action, reward, done = env.step(env.action_space.sample()) # take a random action
    if s==20:
        print(obs)
        print(obs.shape)
        img = obs.reshape(obs.shape[0],obs.shape[1])
        img = Image.fromarray(img, 'L')
        img.show()
