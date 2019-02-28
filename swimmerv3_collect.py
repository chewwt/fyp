import gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

NUM_EPISODES = 5
NAME_ENV = 'Swimmer-v3'
TRAIN_STEPS = 3000000

model_names = ['swimmerv3_r0_ppo2_'+str(TRAIN_STEPS), 'swimmerv3_r1_ppo2_'+str(TRAIN_STEPS), 'swimmerv3_r2_ppo2_'+str(TRAIN_STEPS)]
forward_weights = [1.0, 1.2, 0.8]    # first is default
ctrl_weights = [1e-4, 1e-4, 1e-4]    # first is default

for mi in range(len(model_names)):
    print('Creating an environment with')
    print('    forward_reward_weight:', forward_weights[mi])
    print('         ctrl_cost_weight:', ctrl_weights[mi])
    env = gym.make(NAME_ENV, 
                   forward_reward_weight=forward_weights[mi],
                   ctrl_cost_weight=ctrl_weights[mi],
                   exclude_current_positions_from_observation=True)
    env = env.unwrapped
    # to reproduce results
    # env.seed(1)

    model = PPO2('MlpPolicy', NAME_ENV, tensorboard_log='./' + model_names[mi] + '_tb/')

    print('Learning PPO2 model:', model_names[mi])
    # learning
    model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=model_names[mi])
    model.save(model_names[mi])

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

        print((e+1), "/", NUM_EPISODES, "   Total reward:", float(epi_rewards))

        total_rewards += epi_rewards
                
    print("Mean reward over", NUM_EPISODES," episodes:", float(total_rewards)/NUM_EPISODES)

    env.close()

    # print("RESULTS: ", total_rewards)
    print('---------------------------------------------')