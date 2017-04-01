import numpy as np
import gym

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
import keras.layers.convolutional as convolutional
import keras.layers.pooling as pooling

import rl
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import car_envs


ENV_NAME = 'car-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
# model.add(convolutional.Convolution2D(32, 3, 3, activation='tanh', dim_ordering='th',
#     input_shape=(1,) + env.observation_space.shape))
# model.add(pooling.MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# model.add(convolutional.Convolution2D(32, 3, 3, activation='tanh', dim_ordering='th'))
# model.add(pooling.MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# model.add(convolutional.Convolution2D(32, 3, 3, activation='tanh', dim_ordering='th'))
# model.add(pooling.MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# model.add(convolutional.Convolution2D(16, 3, 3, activation='tanh', dim_ordering='th'))
# model.add(Flatten())
# model.add(Dense(128, activation='tanh'))
model.add(Reshape(env.observation_space.shape, input_shape=(1,) + env.observation_space.shape))
model.add(convolutional.Convolution2D(32, 9, 9, subsample=(4, 4),
    activation='relu', dim_ordering='tf'))
model.add(convolutional.Convolution2D(32, 5, 5, subsample=(2, 2),
    activation='relu', dim_ordering='tf'))
model.add(convolutional.Convolution2D(32, 3, 3,
    activation='relu', dim_ordering='tf'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=5000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=128,
               target_model_update=0.1, policy=policy, batch_size=128)
dqn.compile(keras.optimizers.SGD(lr=0.0001), metrics=['mae'])

callbacks = [rl.callbacks.FileLogger('./log.txt', interval=10)]
# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, callbacks=callbacks, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
