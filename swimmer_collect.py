import gym

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

NUM_EPISODES = 1
NAME_ENV = 'Swimmer-v2'

env = gym.make(NAME_ENV)
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

env = env.unwrapped
# to reproduce results
env.seed(1)

model_name = 'swimmer_r0_ppo2'

model = PPO2('MlpPolicy', NAME_ENV, tensorboard_log='./' + model_name + '_tb/')

# learning
model.learn(total_timesteps=50000, tb_log_name=model_name)
model.save(model_name)

total_rewards = 0.

#-------- run the model -------#
for e in range(NUM_EPISODES):
    obs = env.reset()
    # env.reset()
    epi_rewards = 0.

    for i in range(1000):
        # env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        epi_rewards += reward
        if done:
            break

    if done:
        print("Episode finished after", i+1, "timesteps")
    else:
        print("Episode not finished")
    print("Total reward:", float(epi_rewards))
    print((e+1), "/", NUM_EPISODES)

    total_rewards += epi_rewards
            
print("Mean reward over", NUM_EPISODES," episodes:", float(total_rewards)/NUM_EPISODES)

env.close()

print("RESULTS: ", total_rewards)