import gym

from stable_baselines import PPO2

import numpy as np
import matplotlib.pyplot as plt
import time
import yaml

ACTION_LIMIT_LOW = -20
ACTION_LIMIT_HIGH = 20
NUM_ACTIONS = 10000
# NAME_ENV = 'Swimmer-v3'
# OBS_DIM = 10
# VIEW_OBS_DIM = 10
# ACT_DIM = 2
NAME_ENV = 'Ant-v3'
OBS_DIM = 15+6+8+14*6
VIEW_OBS_DIM = 28
ACT_DIM = 8

env = gym.make(NAME_ENV, exclude_current_positions_from_observation=False)
env = env.unwrapped

obs = env.reset()

states = np.array([]).reshape(0, OBS_DIM)

for i in range(NUM_ACTIONS):

    # action = np.random.uniform(ACTION_LIMIT_LOW, ACTION_LIMIT_HIGH, 2)
    action = np.clip(np.random.normal(0.0, 0.5, ACT_DIM), ACTION_LIMIT_LOW, ACTION_LIMIT_HIGH)
    obs, reward, done, info = env.step(action)
    obs = np.array(obs).reshape(1, OBS_DIM)
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

# with open(NAME_ENV+'_states.yaml', 'w') as f:
#     yaml.dump(states_dist, f)

# swimmer
# titles = ['x', 'y', 'theta1', 'theta2', 'theta3', \
#     'dot_x', 'dot_y', 'dot_theta1', 'dot_theta2', 'dot_theta3']

# ant 
titles = ['x', 'y', 'qx', 'qy', 'qz', 'qw', \
        'theta1', 'theta2', 'theta3', 'theta4', \
        'theta5', 'theta6', 'theta7', 'theta8',
        'dot_x', 'dot_y', 'dot_z', 'omega_x', 'omega_y', 'omega_z',
        'dot_theta1', 'dot_theta2', 'dot_theta3', 'dot_theta4', \
        'dot_theta5', 'dot_theta6', 'dot_theta7', 'dot_theta8']

# 2, 5 ,1
for i in range(VIEW_OBS_DIM):
    print(titles[i], '  max', np.max(states[:, i]), '   min', np.min(states[:,i]))

    plt.subplot(2, VIEW_OBS_DIM/2, i+1)
    plt.hist(states[:, i], bins='auto') 
    plt.title(titles[i])


plt.show()

print(states.shape) 