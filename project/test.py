from __future__ import division
import argparse

from PIL import Image
import numpy as np
import gym
import gym_snake

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import sys

from gym.envs.registration import register
register(
    id='MiniGrid-Empty-8x8-v1',
    entry_point='gym_minigrid.envs:EmptyEnv8x8',
)

register(
    id='MiniGrid-KeyCorridorS6R3-v1',
    entry_point='gym_minigrid.envs:KeyCorridorS6R3',
)





INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        #print('obs',observation.shape)
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='MiniGrid-KeyCorridorS6R3-v1')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('-n', '--n_snakes', type=int, default=1,
                    help='Number of snakes in multi-snake setting')
parser.add_argument('-f', '--n_foods', type=int, default=100,
                    help='Number of foods to use')

args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
"""env.grid_size = [84,84]
env.unit_size = 1
env.unit_gap = 0
env.n_snakes = args.n_snakes
env.n_foods = args.n_foods"""
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# The trade-off between exploration and exploitation is difficult and an on-going research topic.
# If you want, you can experiment with the parameters or use a different policy. Another popular one
# is Boltzmann-style exploration:
# policy = BoltzmannQPolicy(tau=1.)
# Feel free to give it a try!

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

print('SENSE CHECK GAME ENV')
# check observation space
# check action space
# check initial image dimensions , input shape

env_check = gym.make(args.env_name)
img = {}

if "snake" in args.env_name:
    env_check.grid_size = [84,84]
    env_check.unit_size = 1
    env_check.unit_gap = 0
    env_check.n_snakes = args.n_snakes
    env_check.n_foods = args.n_foods
    img['obs'] = env_check.reset()
    print('action_space',env.action_space)
    print('observation_space',env.observation_space)
    for m in range(10):
        env_check.render()
        obs, action, reward, done = env_check.step(env.action_space.sample())
        if m == 5:
            print(obs.shape)
            img['height'] = obs.shape[0]
            img['width'] = obs.shape[1]
            img['channels'] = obs.shape[2]
            # img['obs'] = obs

            print(action)
            print(reward)
else:
    img['obs'] = env_check.reset()
    print('action_space',env.action_space)
    print('observation_space',env.observation_space)
    for m in range(50):
        env_check.render()
        obs, action, reward, done = env_check.step(env.action_space.sample())
        if m == 10:
            print('data shape',obs['image'].shape)
            img['height'] = obs['image'].shape[0]
            img['width'] = obs['image'].shape[1]
            img['channels'] = obs['image'].shape[2]
            img['obs'] = obs['image']
            full_image = env_check.render('rgb_array')
            print('full image',full_image,full_image.shape)
            data = np.zeros((full_image.shape[0], full_image.shape[1], full_image.shape[2]), dtype=np.uint8)
            showimg = Image.fromarray(full_image,'RGB')
            showimg = showimg.resize((84,84),Image.ANTIALIAS)
            #print('new img size',showimg.size())
            showimg.show()

            print('SAR')
            print(obs)
            print(action)
            print(reward)


#print('obs',img['obs'])
#print('obs shape',img['obs'].shape)

full_image = env_check.render('rgb_array')

#data = np.zeros((img['height'], img['width'], img['channels']), dtype=np.uint8)
#showimg = Image.fromarray(img['obs']*255/np.amax(img['obs']),'RGB')
#showimg.show()


#sys.exit(0)


print('SENSE CHECK PREPROCESSING')
# check
print('SENSE CHECK DQN AGENT')
x = dqn.get_config()
for key, value in x.items():
    if key=='model':
        pass
    else:
        print(key,value)



"""
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=10, visualize=False)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)


"""
