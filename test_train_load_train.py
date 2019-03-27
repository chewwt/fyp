import gym

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

NAME_ENV = 'Swimmer-v3'
MODEL_NAME = 'test'

env = gym.make(NAME_ENV, 
               forward_reward_weight=1.0,
               ctrl_cost_weight=1e-4,
               exclude_current_positions_from_observation=False)
# env = env.unwrapped
env = DummyVecEnv([lambda: env])

# to reproduce results
# env.seed(1)

model = PPO2('MlpPolicy', env, tensorboard_log='./test_tb/')

print('Learning PPO2 model')
# learning
print(model.num_timesteps)

model.learn(total_timesteps=10000, tb_log_name='test')
model.save(MODEL_NAME)
print(model.num_timesteps)

del model

print('Learning PPO2 model again')

model = PPO2.load(MODEL_NAME, env=env, tensorboard_log='./test_tb/')
# model.set_env(env)
# model.verbose = 1
# model.num_timesteps
print(model.num_timesteps)

model.learn(total_timesteps=40000, tb_log_name='test' , reset_num_timesteps=False) 
model.save(MODEL_NAME)
print(model.num_timesteps)

