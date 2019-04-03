import numpy as np
from matplotlib import pyplot as pl
import glob

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MAX_ITER = 11

random_tb_files = 'antv3_unclip_20_unhide_random_base*_expert_r0_trajsteps_1_nstates_100_nsamples_50*/r0/*'
# random_tbs = glob.glob(random_tb_files)

mmd_tb_files = 'antv3_unclip_20_unhide_mmd_base*_expert_r0_trajsteps_1_nstates_100_nsamples_50*/r0/*'
# mmd_tbs = glob.glob(mmd_tb_files)

values_r0 = {'random': np.array([]).reshape(-1, MAX_ITER), 'mmd': np.array([]).reshape(-1, MAX_ITER)}
means_r0 = {}
std_r0 = {}

for key, files in zip(['mmd', 'random'], [mmd_tb_files, random_tb_files]):
# for key, files in zip(['mmd'], [mmd_tb_files]):
	tbs = glob.glob(files)
	# print(tbs)

	for tb in tbs:
		# print(tb)
		ea = EventAccumulator(tb)
		ea.Reload()

		# print(ea.Tags())
		# print(ea.Scalars('P_R_'))

		vs = np.array([])

		for i in ea.Scalars('P_R_'):
			vs = np.append(vs, i.value)

		if len(vs) != MAX_ITER:
			continue


		# print(values_r0[key].shape, vs.shape)
		values_r0[key] = np.vstack((values_r0[key], vs))

		# print(values_r0[key])

	means_r0[key] = np.mean(values_r0[key], axis=0)
	std_r0[key] = np.std(values_r0[key], axis=0)

# print(values_r0['mmd'].shape)

t = np.arange(0, MAX_ITER)

pl.title('P(R=r0)')

pl.plot(t, means_r0['mmd'], 'k', color='#CC4F1B', label='MI', alpha=0.7)
pl.fill_between(t, means_r0['mmd']-std_r0['mmd'], means_r0['mmd']+std_r0['mmd'], facecolor='#FF9848', alpha=0.4)

pl.plot(t, means_r0['random'], 'k', color='#077ACC', label='Random', alpha=0.7)
pl.fill_between(t, means_r0['random']-std_r0['random'], means_r0['random']+std_r0['random'], facecolor='#3BDFFF', alpha=0.4)

pl.legend()

pl.show()
