import gym

from stable_baselines import PPO2

import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

ACTION_LIMIT_LOW = -100
ACTION_LIMIT_HIGH = 100
NUM_ACTIONS = 10000
NAME_ENV = 'Swimmer-v3'

env = gym.make(NAME_ENV, exclude_current_positions_from_observation=False)
env = env.unwrapped

obs = env.reset()

states = np.array([]).reshape(0, 10)

for i in range(NUM_ACTIONS):

    # action = np.random.uniform(ACTION_LIMIT_LOW, ACTION_LIMIT_HIGH, 2)
    action = np.clip(np.random.normal(0.0, 0.5, 2), ACTION_LIMIT_LOW, ACTION_LIMIT_HIGH)
    obs, reward, done, info = env.step(action)
    obs = np.array(obs).reshape(1, 10)
    states = np.vstack((states, obs))

# x = states[:, 0]
# y = states[:, 1]
# theta1 = states[:, 2]
# theta2 = states[:, 3]
# theta3 = states[:, 4]

# dot_x = states[:, 5]
# dot_y = states[:, 6]
# dot_theta1 = states[:, 7]
# dot_theta2 = states[:, 8]
# dot_theta3 = states[:, 9]

# cov = np.cov(states, rowvar=False)
# print(cov)
# print(cov.shape)
# mean = np.mean(states, axis=0)
# print(mean)
# print(mean.shape)

# states_dist = {'mean': mean.tolist(), 'cov': cov.tolist(), 'n_dims': 10}

# with open('states.yaml', 'w') as f:
#     yaml.dump(states_dist, f)


titles = ['x', 'y', 'theta1', 'theta2', 'theta3', \
    'dot_x', 'dot_y', 'dot_theta1', 'dot_theta2', 'dot_theta3']

# 2, 5 ,1
for i in range(10):
    print(titles[i], '  max', np.max(states[:, i]), '   min', np.min(states[:,i]))

    plt.subplot(2, 5, i+1)
    plt.hist(states[:, i], bins='auto') 
    plt.title(titles[i])


plt.show()

print(states.shape) 