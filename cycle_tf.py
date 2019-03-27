import gym
from stable_baselines import PPO2

import argparse
import yaml
import numpy as np
from scipy.stats import norm, multivariate_normal

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions

NAME_ENV = 'Swimmer-v3'
ACTION_DIMS = 2
HIDE_OBS_DIMS = 8
UNHIDE_OBS_DIMS = 10
OBS_DIMS = None

N_POLICIES = 3

MAX_ITER = 2
N_STATES = 500
TRAJ_STEPS = 2

SIGMA = 7.0

# number of samples to model a distribution (for mmd) [too high will run out of memory]
N_SAMPLES = 100

FLOAT_MIN = -700

class CycleTf():

    def __init__(self):
        pass


    #=========  Setup  =========#
    def setup(self, model_names, expert_model, logdir='logs', measure='mvnkl', unhide=False):
        # env = gym.make(NAME_ENV)
        # env = env.unwrapped

        # print("high: ", env.action_space.high)
        # print("low: ", env.action_space.low)
        # to reproduce results
        # env.seed(1)
        global OBS_DIMS
        if unhide:
            OBS_DIMS = UNHIDE_OBS_DIMS
        else:
            OBS_DIMS = HIDE_OBS_DIMS

        #------ load the models --------#
        self.model_names = model_names
        self.models = self.load_models(model_names)  # actual PPO2 models
        self.expert_model = self.load_models(expert_model)

        #------ make environment --------#
        self.env = gym.make(NAME_ENV, exclude_current_positions_from_observation=(not unhide))
        self.env = self.env.unwrapped

        #------ init reward beliefs --------#    
        self.prob_rs = np.array(N_POLICIES * [1./N_POLICIES])
        self.log_prob_rs = np.log(self.prob_rs)

        self.measure = measure

        self.build_tf_graph(measure=measure)
        self.setup_tensorboard(logdir, measure=measure)

        self.sess = tf.Session()

        print('Setup done')


    # return a PPO2 model / an array of PPO2 models according to the model names
    def load_models(self, model_names):

        if type(model_names) == list:
            models = []
            print('Loading models...')

            for i in range(len(model_names)):
                models.append(PPO2.load(model_names[i]))

            print('Loading done')

            return models
        else:
            return PPO2.load(model_names)


    def build_tf_graph(self, measure='mvnkl'):

        print('Building tf graph')

        # action probability distributions for each N_STATES state for the different policies
        self.mu_raw_ph = {}
        # self.sigma_ph = {}
        self.cov_ph = {}

        for i in range(N_POLICIES):
            self.mu_raw_ph[i] = tf.placeholder(tf.float32, shape=(N_STATES, ACTION_DIMS), name='mu'+str(i))
            # self.sigma_ph[i] = tf.placeholder(tf.float32, shape=(N_STATES, ACTION_DIMS), name='sigma'+str(i))
            self.cov_ph[i] = tf.placeholder(tf.float32, shape=(N_STATES, ACTION_DIMS, ACTION_DIMS), name='cov'+str(i))
        
        # print(self.sigma_ph)

        # reward functions probabilities
        self.norm_prob_rs_ph = tf.placeholder(tf.float32, shape=(N_POLICIES), name='norm_Pr')

        self.prob_r0, self.prob_r1, self.prob_r2 = tf.split(self.norm_prob_rs_ph, num_or_size_splits=3, axis=0)
        self.prob_r0 = tf.reshape(self.prob_r0, [])
        self.prob_r1 = tf.reshape(self.prob_r1, [])
        self.prob_r2 = tf.reshape(self.prob_r2, [])
        
        # With reference from https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
        # Quadratic-time MMD with Gaussian RBF kernel
        # X, Y are np arrays of samples, with shapes (N_STATES, ACTION_DIMS)
        if measure == 'mmd':

            self.mmd_sigma_ph = tf.placeholder(tf.float32, shape=(), name='mmd_sigma')
            self.mmd_gamma = 1 / (2 * tf.square(self.mmd_sigma_ph))

            # Sampling
            self.mvn = {}
            self.samples = {}
            self.samplesT = {}
            # self.
            for i in range(N_POLICIES):
                # self.mvn[i] = tfd.MultivariateNormalFullCovariance(loc=self.mu_raw_ph[i], covariance_matrix=self.cov_ph[i])
                sigma = tf.sqrt(tf.linalg.diag_part(self.cov_ph[i]))
                self.mvn[i] = tfd.MultivariateNormalDiag(loc=self.mu_raw_ph[i], scale_diag=sigma)
                print(self.mvn[i])
                self.samples[i] = tf.transpose(self.mvn[i].sample(N_SAMPLES), perm=[1,0,2])
                print(self.samples[i])
                self.samplesT[i] = tf.transpose(self.samples[i], perm=[0,2,1])
                print(self.samplesT[i])
                print('----')

            print(tf.reshape(tf.tile(self.norm_prob_rs_ph, [N_STATES]), (N_STATES, N_POLICIES)))

            # Average of the distributions
            self.mvn['avg'] = tfd.Mixture(
                                    cat=tfd.Categorical(probs=tf.reshape(tf.tile(self.norm_prob_rs_ph, [N_STATES]), (N_STATES, N_POLICIES))), \
                                    # cat=tfd.Categorical(probs=[0.3, 0.3, 0.4]), \
                                    components=[self.mvn[0], self.mvn[1], self.mvn[2]])

            # self.mvn['avg'] = tfd.Mixture(
            #                         cat=tfd.Categorical(probs=[0.3, 0.3, 0.4]), \
            #                         components=[
            #                             tfd.MultivariateNormalDiag(loc=[-6, 4], scale_diag=[4, 0.1]),
            #                             tfd.MultivariateNormalDiag(loc=[2, -2], scale_diag=[8, 0.1]),
            #                             tfd.MultivariateNormalDiag(loc=[10, 2], scale_diag=[6, 0.05])
            #                         ])

            
            print(self.mvn['avg'])

            self.samples['avg'] = tf.transpose(self.mvn['avg'].sample(N_SAMPLES), perm=[1,0,2])

            print('samples', self.samples['avg'])
            self.samplesT['avg'] = tf.transpose(self.samples['avg'], perm=[0,2,1])
            print('samplesT', self.samplesT['avg'])

            # Calculate with Kernel trick?
            self.mmd_mul = {}
            self.mmd_sqnorms = {}
            self.mmd_K = {}

            self.mmd_mul['avg_avg'] = tf.matmul(self.samples['avg'], self.samplesT['avg'])
            # print('mmd_mul', self.mmd_mul['avg_avg'])
            self.mmd_sqnorms['avg'] = tf.linalg.diag_part(self.mmd_mul['avg_avg'])
            # print('mmd_sqnorms', self.mmd_sqnorms['avg'])
            # print(2 * tf.expand_dims(self.mmd_sqnorms['avg'], axis=1))
            # print(-2 * self.mmd_mul['avg_avg'] + 2 * tf.expand_dims(self.mmd_sqnorms['avg'], axis=1))
            # print(self.mmd_gamma)
            # print(tf.scalar_mul(-self.mmd_gamma, (-2 * self.mmd_mul['avg_avg'] + 2 * tf.expand_dims(self.mmd_sqnorms['avg'], axis=1))))
            self.mmd_K['avg_avg'] = tf.exp(-self.mmd_gamma * (
                                    -2 * self.mmd_mul['avg_avg'] + 2 * tf.expand_dims(self.mmd_sqnorms['avg'], axis=1)))
            print('mmd_K', self.mmd_K['avg_avg'])
            self.raw_dists = []

    
            for i in range(N_POLICIES):
                
                self.mmd_mul[str(i)+'_avg'] = tf.matmul(self.samples[i], self.samplesT['avg'])
                self.mmd_mul[str(i)+'_'+str(i)] = tf.matmul(self.samples[i], self.samplesT[i])
                self.mmd_sqnorms[i] = tf.linalg.diag_part(self.mmd_mul[str(i)+'_'+str(i)])

                self.mmd_K[str(i)+'_'+str(i)] = tf.exp(-self.mmd_gamma * (
                                                -2 * self.mmd_mul[str(i)+'_'+str(i)] + 2 * tf.expand_dims(self.mmd_sqnorms[i], axis=1)))
                self.mmd_K[str(i)+'_avg'] = tf.exp(-self.mmd_gamma * (
                                                -2 * self.mmd_mul[str(i)+'_avg'] + tf.expand_dims(self.mmd_sqnorms[i], axis=1) + \
                                                tf.expand_dims(self.mmd_sqnorms['avg'], axis=1)))
        
                d = tf.reduce_mean(self.mmd_K[str(i)+'_'+str(i)], axis=[1,2]) + tf.reduce_mean(self.mmd_K['avg_avg'], axis=[1,2]) \
                    - 2 * tf.reduce_mean(self.mmd_K[str(i)+'_avg'], axis=[1,2])
                print(d)
                self.raw_dists.append(d)

            self.mmd_dists = tf.reduce_sum(self.raw_dists, axis=0)
            self.best_index = tf.argmax(self.mmd_dists)

            print('mmd_Ks', self.mmd_K)
            print('dists', self.raw_dists)
            print('mmd_dists', self.mmd_dists)
            print('best_index', self.best_index)
        # KL divergence
        #if measure == 'mvnkl':
        else:
            self.mu = {}

            for i in range(N_POLICIES):
                self.mu[i] = tf.expand_dims(self.mu_raw_ph[i], axis=2)

            self.kls = {}

            for i in range(N_POLICIES):
                for j in range(N_POLICIES):

                    if i == j:
                        continue
     
                    # KL multivariate closed form with weighting
                    self.kls[str(i)+str(j)] = 0.5 * (tf.log(tf.linalg.det(self.cov_ph[j]) / tf.linalg.det(self.cov_ph[i])) + \
                                                tf.linalg.trace(tf.matmul(tf.linalg.inv(self.cov_ph[j]), self.cov_ph[i])) + \
                                                tf.squeeze(tf.matmul(tf.matmul(tf.transpose(self.mu[j] - self.mu[i], perm=[0,2,1]), \
                                                tf.linalg.inv(self.cov_ph[j])), (self.mu[j] - self.mu[i]))) - \
                                                ACTION_DIMS) * (self.norm_prob_rs_ph[i] + self.norm_prob_rs_ph[j])

            self.kl_dists = tf.reduce_sum(tf.stack(list(self.kls.values())), axis=0)  # stack gives (6, N_STATES) tensor, reduce_sum to (N_STATES) tensor
            # print(self.kl_dists)
            self.best_index = tf.argmax(self.kl_dists)

        # print('Done building tf graph')

    def setup_tensorboard(self, logdir='logs', measure='mvnkl'):
        print('Setting up tensorboard')

        # self.r0_writer = tf.summary.FileWriter(logdir + '/r0')
        # self.r1_writer = tf.summary.FileWriter(logdir + '/r1')
        # self.r2_writer = tf.summary.FileWriter(logdir + '/r2')
         
        self.writer = tf.summary.FileWriter(logdir)

        scalar_pr0 = tf.summary.scalar('Reward_Pr0', self.prob_r0)
        scalar_pr1 = tf.summary.scalar('Reward_Pr1', self.prob_r1)
        scalar_pr2 = tf.summary.scalar('Reward_Pr2', self.prob_r2)
        self.summary_pr_op = tf.summary.merge([scalar_pr0, scalar_pr1, scalar_pr2])

        if measure == 'mmd':
            # pass
            hist_dist = tf.summary.histogram('MMD_Distances', self.mmd_dists)
        # if measure == 'mvnkl':
        else:
            # hist_pr = tf.summary.histogram('Reward_Probabilities', self.prob_rs)
            hist_dist = tf.summary.histogram('KL_Distances', self.kl_dists)

        self.summary_dist_op = tf.summary.merge([hist_dist])
        # self.summary_pr_op = tf.summary.merge([hist_pr])
        # self.summary_op = tf.summary.merge_all()


    def cleanup(self):
        if hasattr(self, 'sess'):
            self.sess.close()

    #=================================#

    # get a random state, return qpos, qvel
    # compared visually to obtained states
    def get_random_state(self, yaml_file=None):
        if yaml_file is None:
            qpos = np.array([0., 0., 0., 0., 0.])
            qpos[:2] = np.random.uniform(-2, 5, 2)
            qpos[2] = np.random.uniform(-1.7, 2.2)
            qpos[3:] = np.random.uniform(-1.7, 1.7, 2)

            qvel = np.array([0., 0., 0., 0., 0.])
            qvel[:2] = np.random.normal(0, 0.7, 2)
            qvel[2] = np.random.normal(0, 1.1)
            qvel[3:] = np.random.normal(0, 2.2, 2)

        else:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)

            mean = np.array(data['mean'])
            cov = np.array(data['cov'])

            state = np.random.multivariate_normal(mean, cov)
            qpos = state[:5]
            qvel = state[5:]

        return qpos, qvel


    # takes in starting state, expert's policy model
    # num steps to move given by global variable TRAJ_STEPS
    # returns tau, represented by [a0, a1, ... a(N-1)], [s1, s2, ... aN], where N = steps
    def get_traj(self, obs, piE):
        # env.set_state(s0[0], s0[1])
        # obs = env._get_obs()
        # obs = obs.reshape((-1,8))
        # obs = np.array(obs)

        actions = []
        states = []

        for i in range(TRAJ_STEPS):
            a, _ = piE.predict(obs)
            obs, reward, done, _ = self.env.step(a)
            # print(obs)
            actions.append(a)
            states.append(obs)

            # print('next', s_next)
            # s0 = s_next

        return actions, states


    # TODO try the action_probability function
    def update_r_belief(self, obs, actions_E, states_E):
        
        llh_trajs = np.array([0.] * N_POLICIES)
        # temp = np.array([1.] * N_POLICIES)
        print('updating r belief...')
        for a, next_obs in zip(actions_E, states_E):
            for i,m in enumerate(self.models):

                # MULTIVARIATE NORMAL
                mus, sigmas = m.action_probability(obs) # [[mu1, mu2]], [[sigma1, sigma2]]
                cov = np.diag(np.squeeze(sigmas))
                a = np.squeeze(a)
                log_prob = multivariate_normal.logpdf(a, np.squeeze(mus), cov)
                prob = multivariate_normal.pdf(a, np.squeeze(mus), cov)
                # ONLY ACTION DIM 1 compared
                # p = m.action_probability(obs) # [[mu1, mu2], [sigma1, sigma2]]
                # p = np.squeeze(p)
                # a = np.squeeze(a)
                # print(p)
                # print(a)
                # log_p1 = norm(p[0][0], p[1][0]).logpdf(a[0])
                # p2 = norm(p[0][1], p[1][1]).pdf(a[1])
                # print(mus, cov)
                # print(p)

                # temp[i] *= p1

                print(i, np.exp(log_prob))

                # llh_trajs[i] *= p1 * p2
                llh_trajs[i] += log_prob
                # llh_trajs[i] += log_p1

            obs = [next_obs]

        print(llh_trajs)
        self.log_prob_rs += llh_trajs
        print('after update', self.log_prob_rs)

        # clipping
        self.log_prob_rs[self.log_prob_rs < FLOAT_MIN] = FLOAT_MIN
        print('after clipping', self.log_prob_rs)

        # normalizing
        self.prob_rs = np.exp(self.log_prob_rs)
        # print(self.prob_rs)
        self.prob_rs /= np.sum(self.prob_rs)
        self.log_prob_rs = np.log(self.prob_rs)
        
        print('after normalized', self.log_prob_rs, self.prob_rs)
        # print(np.exp(self.log_prob_rs))
        # print(prob_rs)


    # init a dict of empty arrays to be action probs
    def action_probs_init(self):
        a = {}

        for i in range(N_POLICIES):
            a['mu'+str(i)] = np.array([]).reshape(0, ACTION_DIMS)
            a['cov'+str(i)] = np.array([]).reshape(0, ACTION_DIMS, ACTION_DIMS)
            # a['sigma'+str(i)] = np.array([]).reshape(0, ACTION_DIMS)

        return a


    #================= Cycle ===================#

    def new_cycle(self):
        obs = self.env.reset()

        for i in range(MAX_ITER):
            states = []
            action_probs = self.action_probs_init()
            feeddict = {}
            
            for si in range(N_STATES):
                new_qpos, new_qvel = self.get_random_state(yaml_file='states.yaml')
                self.env.set_state(new_qpos, new_qvel)
                states.append([new_qpos, new_qvel])

                obs = self.env._get_obs()
                obs = obs.reshape((-1,OBS_DIMS))
                # print(obs.shape)
                # obs = np.array(obs)

                ###---- get action probs ----###
                
                for mi, m in enumerate(self.models):
                    # action_probability returns [[mu1, mu2]], [[sigma1, sigma2]] (2D action space)
                    mus, sigmas = m.action_probability(obs)
                    action_probs['mu'+str(mi)] = np.vstack((action_probs['mu'+str(mi)], mus))
                    cov = np.square(np.diag(np.squeeze(sigmas)))
                    action_probs['cov'+str(mi)] = np.vstack((action_probs['cov'+str(mi)], np.expand_dims(cov, axis=0)))
                    # action_probs['sigma'+str(mi)] = np.vstack((action_probs['sigma'+str(mi)], sigmas))

            # set up feeddict
            for pi in range(N_POLICIES):
                feeddict[self.mu_raw_ph[pi]] = action_probs['mu'+str(pi)]
                feeddict[self.cov_ph[pi]] = action_probs['cov'+str(pi)]
                # feeddict[self.sigma_ph[pi]] = action_probs['sigma'+str(pi)]

            feeddict[self.norm_prob_rs_ph] = self.prob_rs

            if self.measure == 'mmd':
                # feeddict[self.mmd_sigma_ph] = 1.0
                feeddict[self.mmd_sigma_ph] = SIGMA

            # print(feeddict)
            # samples_avg = self.sess.run(self.samples['avg'],
            #                                         feed_dict=feeddict)
            # print(samples_avg.shape)
            
            best_index, dists, summary_dist, summary_pr = self.sess.run([self.best_index, self.mmd_dists, self.summary_dist_op, self.summary_pr_op],
                                                    feed_dict=feeddict)
            # best_index, mmd_dists, raw_dists, mmd_K, mmd_mul, mmd_sqnorms, mmd_gamma = self.sess.run([self.best_index, self.mmd_dists, self.raw_dists, self.mmd_K, self.mmd_mul, self.mmd_sqnorms, self.mmd_gamma],
            #                                         feed_dict=feeddict)


            # print(self.prob_rs, self.log_prob_rs)
            # print(kld)
            # print(mmd_dists, raw_dists)
            # print('mmd_K', mmd_K['avg_avg'])
            # print('mmd_mul', mmd_mul['avg_avg'])
            # print('mmd_sqnorms', mmd_sqnorms['avg'])
            # print('mmd_gamma', mmd_gamma)
            print(best_index)

            # summary_dist, summary_pr = self.sess.run([self.summary_dist_op, self.summary_pr_op],
            #                                         feed_dict=feeddict)

            self.writer.add_summary(summary_dist, i+1)
            self.writer.add_summary(summary_pr, i)

            ###---- Get from Expert's Trajectory ----###
            best_state = states[best_index]
            self.env.set_state(best_state[0], best_state[1])
            obs = self.env._get_obs()
            obs = obs.reshape((-1,OBS_DIMS))

            actions_E, states_E = self.get_traj(obs, self.expert_model)
            print(actions_E, states_E)
            self.update_r_belief(obs, actions_E, states_E)

            # print(self.log_prob_rs)
            # self.log_prob_rs

            # p_prob_rs = np.exp(prob_rs)
            # p_prob_rs /= np.sum(p_prob_rs)
            # print('P(R)s: ', p_prob_rs)
            # prob_rs_history = np.vstack((prob_rs_history, p_prob_rs))

        summary_pr = self.sess.run(self.summary_pr_op, \
                            feed_dict={self.norm_prob_rs_ph: self.prob_rs})

        self.writer.add_summary(summary_pr, MAX_ITER)



def main(logdir, measure, unhide):
    # model_names = ['swimmer_r0_ppo2_3000000', 'swimmer_r1new_ppo2_3000000', 'swimmer_r2_ppo2_3000000']
    # model_names = ['swimmerv3_r0_ppo2_3000000', 'swimmerv3_r1_ppo2_3000000', 'swimmerv3_r2_ppo2_3000000']
    # model_names = ['swimmerv3_unclip_unhide_r0_ppo2_3000000', 'swimmerv3_unclip_unhide_r1_ppo2_3000000', 'swimmerv3_unclip_unhide_r2_ppo2_3000000']
    model_names = ['swimmerv3_unclip_unhide_r0_fwd_w_1-0_ctrl_w_0-0001_ppo2_5000000', 'swimmerv3_unclip_unhide_r1_fwd_w_1-005_ctrl_w_0-0001_ppo2_5000000', 'swimmerv3_unclip_unhide_r2_fwd_w_0-995_ctrl_w_0-0001_ppo2_5000000']
    # expert_model = 'swimmerv3_unclip_unhide_r0_ppo2_3000000'
    # expert_model = 'swimmerv3_unclip_unhide_r1_fwd_w_1-05_ctrl_w_0-0001_ppo2_5000000'
    expert_model = 'swimmerv3_unclip_unhide_r0_fwd_w_1-0_ctrl_w_0-0001_ppo2_5000000'

    
    cycle = CycleTf()
    cycle.setup(model_names, expert_model, logdir=logdir, measure=measure, unhide=unhide)
    # prob_rs_history = cycle.cycle()
    # cycle.plot_prob_rs(prob_rs_history)
    cycle.new_cycle()
    cycle.cleanup()

if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Run Active Learning for IRL cycle using stochastic stable baselines models on Swimmer-v3 from OpenAI Gym')
    p.add_argument('--logdir', type=str, default='logs', help='directory to save tensorboard summaries to')
    p.add_argument('--measure', type=str, default='mvnkl', help='distance measure to compare distributions with. mvnkl or mmd')
    p.add_argument('--unhide', type=bool, default=False, help='whether to include all states for observations')

    args = p.parse_args()

    main(args.logdir, args.measure, args.unhide)
    
