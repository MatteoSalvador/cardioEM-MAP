#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pyCirculationANN as pyC

params_file             = 'params/params_bsln.json'
initstate_file          = 'states/initstate_bsln.json'
filename_output         = 'circulation_ANN'
out_folder              = 'simulations'

picturename_pv_loops    = out_folder + '/circulation_pv_loop_ANN.png'
picturename_elastances  = out_folder + '/circulation_elastances_ANN.png'
picturename_variables   = out_folder + '/circulation_ANN.png'

circ = pyC.circulation_closed_loop(test_case = 'none', options = params_file, params = False)
print('******* Run ANN-0D closed-loop circulation model *******')
circ.run(num_cycles = 5, dt = 1e-2, initial_state = initstate_file, output_folder = out_folder, filename_output = filename_output, last_HB = False)
			
history = pd.read_csv(out_folder + '/' + filename_output + '_data.csv', delimiter = ',').apply(pd.to_numeric, errors = 'coerce')

pyC.postprocessing.plot_PV_loops(history)
pyC.postprocessing.plt.savefig(picturename_pv_loops)
pyC.postprocessing.plot_elastances(history)
pyC.postprocessing.plt.savefig(picturename_elastances)
pyC.postprocessing.plot_history(history)
pyC.postprocessing.plt.savefig(picturename_variables)

print('Plots saved!')
