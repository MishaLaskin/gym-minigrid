"""
Use DQN to train a model on Atari environments.
For example, train on Pong like:
    $ python atari_dqn.py Pong
"""

import argparse
from functools import partial
import sys
import cv2
from PIL import Image

from anyrl.algos import DQN
from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import BatchedFrameStack, DownsampleEnv, GrayscaleEnv
from anyrl.models import NatureQNetwork, EpsGreedyQNetwork, rainbow_models
from anyrl.rollouts import BatchedPlayer, UniformReplayBuffer
from anyrl.spaces import gym_space_vectorizer
import gym
from gym.envs.registration import register
from gym.spaces import Box
import tensorflow as tf
import numpy as np
from gym_minigrid.wrappers import FullyObsWrapper

REWARD_HISTORY = 1


register(
    id='MiniGrid-Empty-8x8-v1',
    entry_point='gym_minigrid.envs:EmptyEnv',
)

register(
    id='MiniGrid-KeyCorridorS6R3-v1',
    entry_point='gym_minigrid.envs:KeyCorridorS6R3',
)


def main():
    """
    Entry-point for the program.
    """
    args = _parse_args()

    # batched env = creates gym env, not sure what batched means
    # make_single_env = GrayscaleEnv > DownsampleEnv
    # GrayscaleEnv = turns RGB into grayscale
    # DownsampleEnv = down samples observation by N times where N is the specified variable (e.g. 2x smaller)
    env = batched_gym_env([partial(make_single_env, args.game)] * args.workers)
    env_test = make_single_env(args.game)
    #make_single_env(args.game)
    print('OBSSSS',env_test.observation_space)
    #env = CustomWrapper(args.game)
    # Using BatchedFrameStack with concat=False is more
    # memory efficient than other stacking options.
    env = BatchedFrameStack(env, num_images=4, concat=False)

    with tf.Session() as sess:
        def make_net(name):
            return rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200)
        dqn = DQN(*rainbow_models(sess,
                              env.action_space.n,
                              gym_space_vectorizer(env.observation_space),
                              min_val=-200,
                              max_val=200))
        player = BatchedPlayer(env, EpsGreedyQNetwork(dqn.online_net, args.epsilon))
        optimize = dqn.optimize(learning_rate=args.lr)

        sess.run(tf.global_variables_initializer())

        reward_hist = []
        total_steps = 0

        def _handle_ep(steps, rew):
            nonlocal total_steps
            total_steps += steps
            reward_hist.append(rew)
            if len(reward_hist) == REWARD_HISTORY:
                print('%d steps: mean=%f' % (total_steps, sum(reward_hist) / len(reward_hist)))
                reward_hist.clear()

        dqn.train(num_steps=int(1e7),
                  player=player,
                  replay_buffer=UniformReplayBuffer(args.buffer_size),
                  optimize_op=optimize,
                  target_interval=args.target_interval,
                  batch_size=args.batch_size,
                  min_buffer_size=args.min_buffer_size,
                  handle_ep=_handle_ep)

    env.close()


def make_single_env(game):
    """Make a preprocessed gym.Env."""
    if 'MiniGrid' in game:
        env = PreprocessEnv(FullyObsWrapper(gym.make(game)))
    else:
        env = gym.make(game)

    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    print('action space: %s obs space: %s' % (env.action_space, env.observation_space))
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    #sys.exit()

    return GrayscaleEnv(DownsampleEnv(env, 2))

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
        observation = self.grayscale_2_onechannel(observation)
        return observation

    def grayscale_2_onechannel(self,img):
        height, width = img.shape[0], img.shape[1]
        img = np.zeros((height,width))
        nchannels = 1
        img = np.resize(img, (height, width, nchannels))
        print('RESIZED: %s' % (img.shape))
        return img

    def to_grayscale(self,img):
        img = Image.fromarray(img, 'RGB').convert('L')
        img = np.array(img)
        return img


    def downsample(self,img,rate):
        return img[::rate, ::rate]




def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', help='Adam learning rate', type=float, default=6.25e-5)
    parser.add_argument('--min-buffer-size', help='replay buffer size before training',
                        type=int, default=2000)
    parser.add_argument('--buffer-size', help='replay buffer size', type=int, default=50000)
    parser.add_argument('--workers', help='number of parallel envs', type=int, default=8)
    parser.add_argument('--target-interval', help='training iters per log', type=int, default=8192)
    parser.add_argument('--batch-size', help='SGD batch size', type=int, default=32)
    parser.add_argument('--epsilon', help='initial epsilon', type=float, default=0.1)
    parser.add_argument('--game', help='game name', default='Breakout-v0')
    return parser.parse_args()


if __name__ == '__main__':
    main()
