import gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

import time

NUM_EPISODES = 1
NAME_ENV = 'Swimmer-v2'

env = gym.make(NAME_ENV)

env = env.unwrapped
# to reproduce results
env.seed(1)

# model_name = 'swimmer_r0_ppo2'

# model = PPO2.load(model_name)

obs = env.reset()
print(obs)
# old_state = env.sim.get_state()
# print(old_state)
env.render()

for i in range(-20, 20):
    i /= 5
    old_state = env.sim.get_state()
    # print(old_state)
    ### test qpos
    new_qpos = old_state.qpos.copy()
    # new_qpos[2] = i
    new_qpos[0] = 0
    new_qpos[1] = 0
    new_qpos[2] = 2.8
    new_qpos[3] = 1.0
    new_qpos[4] = -1.4
    env.set_state(new_qpos, old_state.qvel)

    ### test qvel (cannot tell)
    # new_qvel = old_state.qvel.copy()
    # new_qvel[4] = i
    # env.set_state(old_state.qpos, new_qvel)
    
    print(env._get_obs())
    env.render()
    time.sleep(0.2)

# total_rewards = 0.

#-------- run the model -------#
# for e in range(NUM_EPISODES):
#     obs = env.reset()
#     # env.reset()
#     epi_rewards = 0.

#     for i in range(1000):
#         # env.render()
#         action, _states = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         epi_rewards += reward
#         if done:
#             break

#     if done:
#         print("Episode finished after", i+1, "timesteps")
#     else:
#         print("Episode not finished")
#     print("Total reward:", float(epi_rewards))
#     print((e+1), "/", NUM_EPISODES)

#     total_rewards += epi_rewards
            
# print("Mean reward over", NUM_EPISODES," episodes:", float(total_rewards)/NUM_EPISODES)

# env.close()

# print("RESULTS: ", total_rewards)