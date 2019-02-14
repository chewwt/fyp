import gym

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG

NUM_EPISODES = 10
NAME_ENV = 'MountainCarContinuous-v0'

env = gym.make(NAME_ENV)
# env = gym.make('CartPole-v1')
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

env = env.unwrapped
# to reproduce results
env.seed(1)

#------- train the model ------#

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)
# model = PPO2('MlpPolicy', 'CartPole-v1').learn(10000)
# model = PPO2('MlpPolicy', NAME_ENV, verbose=1, tensorboard_log="./ppo_car/")
# model = PPO2('MlpPolicy', NAME_ENV, tensorboard_log="./ppo_car_tb/")
# model.learn(total_timesteps=10000, tb_log_name="first_run")
# model.learn(total_timesteps=10000, tb_log_name="second_run")
# model.learn(total_timesteps=10000, tb_log_name="third_run")

models = ["a2c_car", "ppo2_car", "ddpg_car"]
types = [A2C, PPO2, DDPG]

for i in range(3):
    print("Training model", models)
    model = types[i]('MlpPolicy', NAME_ENV, tensorboard_log="./comp_car_tb/")
    # model = types[i].load(models[i])
    model.learn(total_timesteps=50000, tb_log_name=models[i])
    model.save(models[i])

total_rewards = [0., 0., 0.]

for mi, m in enumerate(models):
    del model
    model = types[mi].load(m)
    print("Model: ", m)

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

        total_rewards[mi] += epi_rewards
                
    print("Mean reward over", NUM_EPISODES," episodes:", float(total_rewards[mi])/NUM_EPISODES)

env.close()

print("RESULTS: ")
for i,m in enumerate(models):
    print(m, ":", total_rewards[i])