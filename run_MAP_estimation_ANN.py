#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyCirculationANN as pyC
import MAP

# For reproducibility among multiple runs.
np.random.seed(0)
noise_std   = 10.0
noise_trace = np.random.normal(0.0, noise_std, 80)

# Number of trials for MAP estimation.
n_trials = 10

# {'LV', 'all', 'atria'}.
test_case = 'LV'
# {'QoIs/estimation_LV.json', 'QoIs/estimation_all.json'}
params_estimation_file = 'params/estimation_' + test_case + '.json'
# {'QoIs/weights_LV.json', 'QoIs/weights_all.json'}
QoIs_weights_file = 'QoIs/weights_'+ test_case + '.json'

# Target of parameters.
if (test_case == 'atria'):
	params_target_file = 'params/params_target_atria.json'
else:
	params_target_file = 'params/params_target_LV_all.json'

# Parameters and initial conditions.
params_file     = 'params/params_bsln.json'
initstate_file  = 'states/initstate_bsln.json'

# Output.
filename_output = 'circulation_ANN_MAP'
out_folder      = 'simulations'

idx_trials  = range(n_trials)
opt_times   = []
l2_errors   = []
nits        = []
losses_opt  = []
for index in range(n_trials):
	# Initialize ANN-0D model.
	circ = pyC.circulation_closed_loop(test_case = test_case, options = params_file, params = True)

	# Initialize MAP estimation.
	par_est = MAP.PE(circ, test_case = test_case, noise_trace = noise_trace, params_est = params_estimation_file, params_target = params_target_file, QoIs_weights = QoIs_weights_file)

	# Run MAP estimation.
	print('******* MAP estimation with jax *******')
	opt_time, l2_error, nit, loss_opt = par_est.run(num_cycles = 5, dt = 1e-2, initial_state = initstate_file, output_folder = out_folder, filename_output = filename_output + '_' + str(index))

	# Collect results.
	opt_times.append(opt_time)
	l2_errors.append(l2_error)
	nits.append(nit)
	losses_opt.append(loss_opt)

# Save results.
dict_data = {'time'     : opt_times,
             'l2_error' : l2_errors,
             'n_iter'   : nits,
             'loss'     : losses_opt}
pd.DataFrame(dict_data).to_csv(out_folder + '/statistics.csv', index = False)

# Plots.
_, axes = plt.subplots(2, 2, figsize = (10, 10))

ax = axes[0,0]
ax.plot(idx_trials, opt_times, 'k', marker = 'o')
ax.set_xlabel('Trial')
ax.set_ylabel(r'Optimization time [s]')

ax = axes[0,1]
ax.plot(idx_trials, l2_errors, 'b', marker = 'o')
ax.set_xlabel('Trial')
ax.set_ylabel(r'Discrete $L^2$ relative error')
ax.set_yscale('log')

ax = axes[1,0]
ax.plot(idx_trials, nits, 'g', marker = 'o')
ax.set_xlabel('Trial')
ax.set_ylabel(r'Number of iterations')

ax = axes[1,1]
ax.plot(idx_trials, losses_opt, 'r', marker = 'o')
ax.set_xlabel('Trial')
ax.set_ylabel('Loss function')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(out_folder + '/statistics.png')