### test mmd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# With reference from https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
# Quadratic-time MMD with Gaussian RBF kernel
# X, Y are np arrays of samples, with shapes (N_STATES, ACTION_DIMS)
def rbf_mmd2(X, Y, sigma=1.0, biased=True):
    # print(X.shape)

    gamma = 1 / (2 * sigma**2)

    XX = np.matmul(X, X.T)
    XY = np.matmul(X, Y.T)
    YY = np.matmul(Y, Y.T)
    
    X_sqnorms = np.diagonal(XX)
    Y_sqnorms = np.diagonal(YY)
    print(X_sqnorms.shape)
    print(X_sqnorms[:, np.newaxis].shape)

    
    K_XY = np.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = np.exp(-gamma * (
            -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = np.exp(-gamma * (
            -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

    print('K_XX', K_XX.shape)
    print('X_sqnorms', X.shape)
    print('----')


    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2


#------------- KL --------------#

# returns kl divergence (symmetric) value
# dist = [mu, sigma]
def unikl(dist1, dist2):
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

# kl for multivariate gaussian
# dist1 = [[mu1, mu2], [sigma1, sigma2]]
def mvnkl(dist1, dist2):
    ACTION_DIMS = 2
    cov1 = np.square(np.diag(dist1[1]))
    cov2 = np.square(np.diag(dist2[1]))
    return 0.5 * (np.log(np.linalg.det(cov2) / np.linalg.det(cov1)) + \
                np.trace(np.matmul(np.linalg.inv(cov2), cov1)) + \
                np.squeeze(np.matmul(np.matmul(np.transpose(dist2[0] - dist1[0]), \
                np.linalg.inv(cov2)), (dist2[0] - dist1[0]))) - \
                ACTION_DIMS)


def get_distance(action_probs, independent=False, method='kl'):

    if method == 'mmd':
        N_SAMPLES = 1000
        cov0 = np.square(np.diag(action_probs[0][1]))
        cov1 = np.square(np.diag(action_probs[1][1]))
        X = np.random.multivariate_normal(action_probs[0][0], cov0, N_SAMPLES)
        Y = np.random.multivariate_normal(action_probs[1][0], cov1, N_SAMPLES)
        return rbf_mmd2(X, Y)

    elif method == 'kl':

        if not independent:
            # print(action_probs[0])
            num_dists = action_probs.shape[0]
            summ = 0

            for i in range(num_dists - 1):
                for j in range(i+1, num_dists):
                    kl = mvnkl(action_probs[i], action_probs[j])
                    summ += kl
            
            return summ
            
        else:
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
                    kl1 = unikl(dists['dim1'][i], dists['dim1'][j])
                    kl2 = unikl(dists['dim2'][i], dists['dim2'][j])

                    # print(i, j, kl1, kl2)

                    summ += kl1 + kl2
                    # summ += kl1

            return summ


def plot_action_probs(action_probs):
    plt.figure()
    color = ['b', 'g', 'r']
    # print('hi')
    # print(action_probs)
    for i, p in enumerate(action_probs):
        a1 = np.arange(-20, 20, 0.01)
        a2 = np.arange(-20, 20, 0.01)
        p = np.squeeze(p)
        print(p)
        print(i,p)
        plt.subplot(1, 2, 1)
        plt.plot(a1, norm.pdf(a1, p[0][0], p[1][0]), color[i])
        plt.subplot(1, 2, 2)
        plt.plot(a2, norm.pdf(a2, p[0][1], p[1][1]), color[i])
        # plt.plot(a1, norm.pdf(a1, p[0], p[1]), color[i])

    plt.show()


def main():
    dist1 = [[-5, -5], [5, 0.1]] # [[mu1, mu2], [sigma1, sigma2]]
    dist2 = [[13, -3], [4, 0.1]]
    dist3 = [[3, -7], [6, 0.08]]

    # dist1 = [[-5], [5]]
    # dist2 = [[13], [4]]
    # dist3 = [[3], [7]]

    dists = np.vstack(([dist1], [dist2]))
    # dists = np.vstack((dists, [dist3]))

    print('unikl', get_distance(dists, independent=True, method='kl'))
    print('mvnkl', get_distance(dists, independent=False, method='kl'))
    print('mmd', get_distance(dists, independent=True, method='mmd'))
    print('--------')
    plot_action_probs(dists)

    dists = np.vstack(([dist1], [dist3]))

    print('unikl', get_distance(dists, independent=True, method='kl'))
    print('mvnkl', get_distance(dists, independent=False, method='kl'))
    print('mmd', get_distance(dists, independent=True, method='mmd'))
    print('--------')
    plot_action_probs(dists)

    dists = np.vstack(([dist2], [dist3]))

    print('unikl', get_distance(dists, independent=True, method='kl'))
    print('mvnkl', get_distance(dists, independent=False, method='kl'))
    print('mmd', get_distance(dists, independent=True, method='mmd'))
    print('--------')
    plot_action_probs(dists)

    dists = np.vstack(([dist1], [dist2]))
    dists = np.vstack((dists, [dist3]))

    plot_action_probs(dists)

if __name__ == '__main__':
    main()
