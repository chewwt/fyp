import gym

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
# from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution

# for plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

MAX_ITER = 100000
NAME_ENV = 'Swimmer-v2'
N_STATES = 10
TRAJ_STEPS = 10

# return an array of PPO2 models according to the model names
def load_models(model_names):
    models = []
    print('Loading models...')

    for i in range(3):
        models.append(PPO2.load(model_names[i]))

    print('Loading done')

    return models


# get a random state, return qpos, qvel
def get_random_state():
    qpos = np.array([0., 0., 0., 0., 0.])
    qpos[:2] = np.random.uniform(-2, 10, 2)
    qpos[2:] = np.random.uniform(-3.14, 3.14, 3)

    qvel = np.random.uniform(-0.1, 0.1, 5)

    return qpos, qvel


# plot the action probs for distributions A,B,C
# action_probs: [[[mu1A, mu2A], [sigma1A, sigma2A], 
#                 [mu1B, mu2B], [sigma1B, sigma2B], ..]]
def plot_action_probs(action_probs):
    plt.figure()
    color = ['b', 'r', 'g']
    # print('hi')
    # print(action_probs)
    for i, p in enumerate(action_probs):
        a1 = np.arange(-20, 20, 0.01)
        a2 = np.arange(-20, 20, 0.01)
        p = np.squeeze(p)
        # print(i,p)
        plt.subplot(1, 2, 1)
        plt.plot(a1, norm.pdf(a1, p[0][0], p[1][0]), color[i])
        plt.subplot(1, 2, 2)
        plt.plot(a2, norm.pdf(a2, p[0][1], p[1][1]), color[i])

    plt.show()


# returns kl divergence (symmetric) value
# dist = [mu, sigma]
def kl(dist1, dist2):
    # print('kl', dist1, dist2)

    d1 = [dist1, dist2]
    d2 = [dist2, dist1]

    # print('hihi', d1, d2)

    summ = 0

    for dA, dB in zip(d1, d2):
        # print(dA, dB)
        mu_A, sigma_A = dA
        mu_B, sigma_B = dB

        kld = np.log(sigma_B / sigma_A) + \
              ((np.square(sigma_A) + np.square(mu_A - mu_B)) / \
              (2 * np.square(sigma_B))) - 0.5

        summ += kld

    return summ


# TODO weight according to reward belief
def get_distance(action_probs):
    dists = {'dim1': [], 'dim2': []}

    for p in action_probs:
        p = np.squeeze(p)
        dists['dim1'].append([p[0][0], p[1][0]]) # [mu1, sigma1]
        dists['dim2'].append([p[0][1], p[1][1]]) # [mu2, sigma2]
        # print('testtest', dists)
    num_dists = len(dists['dim1'])

    summ = 0

    for i in range(num_dists - 1):
        for j in range(i+1, num_dists):
            kl1 = kl(dists['dim1'][i], dists['dim1'][j])
            kl2 = kl(dists['dim2'][i], dists['dim2'][j])

            print(i, j, kl1, kl2)

            # summ += kl1 + kl2
            summ += kl1

    return summ

# def get_distance(action_probs):
#     dists = []
#     kl_sum = 0

#     for p in action_probs:
#         print(p[0])
#         dists.append(DiagGaussianProbabilityDistribution(p[0]))

#     for i in range(len(dists) - 1):
#         for j in range(i+1, len(dists)):
#             kl1 = dists[i].kl(dists[j])
#             kl2 = dists[i].kl(dists[j])
#             kl_sum += kl1 + kl2

#     return kl_sum

# takes in starting state, num steps to move, expert's policy model, gym env
# returns tau, represented by [a0, a1, ... a(N-1)], [s1, s2, ... aN], where N = steps
def get_traj(obs, steps, piE, env):
    # env.set_state(s0[0], s0[1])
    # obs = env._get_obs()
    # obs = obs.reshape((-1,8))
    # obs = np.array(obs)

    actions = []
    states = []

    for i in range(steps):
        a, _ = piE.predict(obs)
        obs, reward, done, _ = env.step(a)
        # print(obs)
        actions.append(a)
        states.append(obs)

        # print('next', s_next)
        # s0 = s_next

    return actions, states


# TODO try the action_probability function
def update_r_belief(obs, actions_E, states_E, models, prob_rs, env):
    
    llh_trajs = np.array([1.] * len(prob_rs))
    print('updating r belief...')
    for a, next_obs in zip(actions_E, states_E):
        for i,m in enumerate(models):

            p = m.action_probability(obs) # [[mu1, mu2], [sigma1, sigma2]]
            p = np.squeeze(p)
            a = np.squeeze(a)
            # print(p)
            # print(a)
            p1 = norm(p[0][0], p[1][0]).pdf(a[0])
            p2 = norm(p[0][1], p[1][1]).pdf(a[1])

            print(p1, p2)

            # llh_trajs[i] *= p1 * p2
            llh_trajs[i] *= p1

        objs = next_obs
        # env.set_state(s[0], s[1])
        # obs = env._get_obs()
        # obs = obs.reshape((-1,8))

    print(llh_trajs)
    prob_rs *= llh_trajs
    # print(prob_rs)


def cycle(models, prob_rs):
    
    env = gym.make(NAME_ENV)
    env = env.unwrapped
    obs = env.reset()
    # print(obs)
    # print(obs.shape)

    for i in range(MAX_ITER):

        ##---- compare different states ----##
        dists = np.zeros(N_STATES)
        states = []

        for si in range(N_STATES):
            new_qpos, new_qvel = get_random_state()
            env.set_state(new_qpos, new_qvel)
            states.append([new_qpos, new_qvel])

            obs = env._get_obs()
            obs = obs.reshape((-1,8))
            # print(obs.shape)
            action_probs = []
            obs = np.array(obs)

            ###---- get action probs ----###
            
            for mi, m in enumerate(models):
                # action_probability returns [[mu1, mu2], [sigma1, sigma2]] (2D action space)
                prob = m.action_probability(obs)
                # print(model_names[mi], prob)
                action_probs.append(prob)
                # action, _ = m.predict(obs)
                # print(action)

            print(action_probs)

            ###---- compare action probs ----###

            dist = get_distance(action_probs)
            print(dist)
            dists[si] = dist

            plot_action_probs(action_probs)

        
        best_index = np.argmax(dists)
        best_state = states[best_index]

        print('Best Index', best_index)
        print('Best State', best_state)
        print('Max Distance', dists[best_index])

        ###---- Get from Expert's Trajectory ----###
        env.set_state(best_state[0], best_state[1])
        obs = env._get_obs()
        obs = obs.reshape((-1,8))

        actions_E, states_E = get_traj(obs, TRAJ_STEPS, models[0], env)
        print(actions_E, states_E)
        update_r_belief(obs, actions_E, states_E, models, prob_rs, env)

        print('P(R)s: ', prob_rs)

        # break


def main():

    # env = gym.make(NAME_ENV)
    # # env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    # env = env.unwrapped

    # print("high: ", env.action_space.high)
    # print("low: ", env.action_space.low)
    # to reproduce results
    # env.seed(1)

    #------ load the models --------#    
    
    model_names = ['swimmer_r0_ppo2_3000000', 'swimmer_r1new_ppo2_3000000', 'swimmer_r2_ppo2_3000000']
    models = load_models(model_names)  # actual PPO2 models

    
    #------ init reward beliefs --------#    
    
    num_models = len(model_names)
    prob_rs = np.array(num_models * [1./num_models])


    #------ start the cycle --------#
    
    cycle(models, prob_rs)
    

if __name__ == '__main__':
    main()
    