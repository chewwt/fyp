# train base r0 to 3m steps, then continue r0,r1,r2 to 4m steps in total
import gym

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

NUM_EPISODES = 5
NAME_ENV = 'Ant-v3'
BASE_TRAIN_STEPS = 800000
TOTAL_TRAIN_STEPS = 1000000
# BASE_TRAIN_STEPS = 20000
# TOTAL_TRAIN_STEPS = 30000

# ctrl_cost_weights = [0.5, 0.505, 0.495]    # first is default
# contact_cost_weights = [5e-4, 5e-4, 5e-4]    # first is default

ctrl_cost_weights = [    0.5, 0.51, 0.49,    0.5,    0.5, 0.52, 0.48, 0.515, 0.485, 0.55, 0.45,     0.5,     0.5,  0.6,  0.4,    0.5,    0.5,  0.505,  0.505,  0.495,  0.495]    # first is default
contact_cost_weights = [5e-4, 5e-4, 5e-4, 5.1e-4, 4.9e-4, 5e-4, 5e-4,  5e-4,  5e-4, 5e-4, 5e-4, 5.05e-4, 4.95e-4, 5e-4, 5e-4, 5.5e-4, 4.5e-4, 5.1e-4, 4.9e-4, 5.1e-4, 4.9e-4]    # first is default

TRAIN_BASE_MODEL = False # set to False if base and r0 from base is already trained. TODO detect the folder autonomatically?

print('Training', len(ctrl_cost_weights) if TRAIN_BASE_MODEL else len(ctrl_cost_weights) - 1, 'models')

model_names = []

model_names.append('antv3_unclip_20_unhide_r0_base2_ctrl_w_' + str(ctrl_cost_weights[0]).replace('.','-') + '_contact_w_' + str(contact_cost_weights[0]).replace('.','-') + '_ppo2_' + str(BASE_TRAIN_STEPS))

for i in range(len(ctrl_cost_weights)):
    if not TRAIN_BASE_MODEL and i == 0:
        continue

    model_names.append('antv3_unclip_20_unhide_r' + str(i) + '_from_base2_' + str(BASE_TRAIN_STEPS) + '_ctrl_w_' + str(ctrl_cost_weights[i]).replace('.','-') + '_contact_w_' + str(contact_cost_weights[i]).replace('.','-') + '_ppo2_' + str(TOTAL_TRAIN_STEPS))


for mi in range(len(model_names)):
    if mi == 0:
        wi = 0    # weight index
    elif TRAIN_BASE_MODEL: 
        wi = mi - 1
    else:
        wi = mi
    print('Creating an ', NAME_ENV, ' environment with')
    print('            ctrl_cost_weight:', ctrl_cost_weights[wi])
    print('         contact_cost_weight:', contact_cost_weights[wi])
    env = gym.make(NAME_ENV, 
                   ctrl_cost_weight=ctrl_cost_weights[wi],
                   contact_cost_weight=contact_cost_weights[wi],
                   exclude_current_positions_from_observation=False)
    # env = env.unwrapped
    env = DummyVecEnv([lambda: env])
    # to reproduce results
    # env.seed(1)

    if mi == 0:    # base policy
        if TRAIN_BASE_MODEL:
            model = PPO2('MlpPolicy', env, tensorboard_log='./' + model_names[mi] + '_tb/')

            print('Learning Base PPO2 model:', model_names[mi])
            # # learning
            model.learn(total_timesteps=BASE_TRAIN_STEPS, tb_log_name=model_names[mi])
            model.save(model_names[mi])
        else:
            continue

    else:
        print('Learning PPO2 model:', model_names[mi])
        model = PPO2.load(model_names[0], env=env, tensorboard_log='./' + model_names[0] + '_tb/')
        model.learn(total_timesteps=TOTAL_TRAIN_STEPS-BASE_TRAIN_STEPS, tb_log_name=model_names[mi] , reset_num_timesteps=False) 
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