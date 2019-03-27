import yaml
import numpy as np
import matplotlib.pyplot as plt

NUM_STATES = 10000

yaml_file = 'states.yaml'

with open(yaml_file, 'r') as f:
    data = yaml.safe_load(f)

mean = np.array(data['mean'])
cov = np.array(data['cov'])

states = np.random.multivariate_normal(mean, cov, 10000)
print(states.shape)

titles = ['x', 'y', 'theta1', 'theta2', 'theta3', \
    'dot_x', 'dot_y', 'dot_theta1', 'dot_theta2', 'dot_theta3']

# 2, 5 ,1
for i in range(10):
    print(titles[i], '  max', np.max(states[:, i]), '   min', np.min(states[:,i]))

    plt.subplot(2, 5, i+1)
    plt.hist(states[:, i], bins='auto') 
    plt.title(titles[i])


plt.show()


