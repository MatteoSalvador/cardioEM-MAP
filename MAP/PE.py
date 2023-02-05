import numpy as np
from numpy import linalg as LA
import pandas as pd
import json
import csv
import time
import math

import scipy.optimize as spopt

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

import jax
import jax_utils
import jax.numpy as jnp
from jax.random import PRNGKey
from jax.config import config
config.update("jax_enable_x64", True)

import sys
sys.path.append('../cardioEM-MAP')
from utils import *

from .postprocessing import *


class PE:

    def __init__(self, circulation_model, test_case, noise_trace, params_est = dict(), params_target = dict(), QoIs_weights = dict()):

        # Get circulation model.
        self.circulation_model = circulation_model

        # Get test case label.
        self.test_case = test_case

        # Parameters for MAP estimation.
        if isinstance(params_est, str):
            with open(params_est, mode='r', newline='') as inputfile:
                params_est = json.loads(inputfile.read())
        params_est_curr           = params_est.get('LA', dict())        
        self.estimate_EB_LA       = bool(params_est_curr.get('EB', 0))     # 0
        params_est_curr           = params_est.get('LV', dict())
        self.estimate_a_XB_LV     = bool(params_est_curr.get('a_XB', 0))   # 1        
        params_est_curr           = params_est.get('RA', dict())
        self.estimate_EB_RA       = bool(params_est_curr.get('EB', 0))     # 2
        params_est_curr           = params_est.get('RV', dict())
        self.estimate_EA_RV       = bool(params_est_curr.get('EA', 0))     # 3
        params_est_curr           = params_est.get('SYS', dict())
        self.estimate_R_AR_SYS    = bool(params_est_curr.get('R_AR', 0))   # 4
        self.estimate_R_VEN_SYS   = bool(params_est_curr.get('R_VEN', 0))  # 5
        self.estimate_V_tot_heart = bool(params_est.get('V_tot_heart', 0)) # 6

        # Loss weights.
        if isinstance(QoIs_weights, str):
            with open(QoIs_weights, mode='r', newline='') as inputfile:
                QoIs_weights = json.loads(inputfile.read())
        QoIs_weights_curr    = QoIs_weights.get('LA', dict())
        self.weight_V_LA     = float(QoIs_weights_curr.get('volume', 0.))
        QoIs_weights_curr    = QoIs_weights.get('RA', dict())
        self.weight_V_RA     = float(QoIs_weights_curr.get('volume', 0.))
        QoIs_weights_curr    = QoIs_weights.get('LV', dict())
        self.weight_V_LV     = float(QoIs_weights_curr.get('volume', 0.))
        QoIs_weights_curr    = QoIs_weights.get('RV', dict())
        self.weight_V_RV     = float(QoIs_weights_curr.get('volume', 0.))
        QoIs_weights_curr    = QoIs_weights.get('ARSYS', dict())
        self.weight_p_AR_SYS = float(QoIs_weights_curr.get('pressure', 0.))

        # Target values for the parameters.
        if isinstance(params_target, str):
            with open(params_target, mode='r', newline='') as inputfile:
                params_target = json.loads(inputfile.read())
        self.params_target  = []
        # EB_LA.
        params_target_curr  = params_target.get('LA', dict())
        self.params_target += [np.array([float(params_target_curr.get('EB', 0.15))])]
        # a_XB.
        params_target_curr  = params_target.get('LV', dict())
        self.params_target += [np.array([float(params_target_curr.get('a_XB', 250.0))])]
        # EB_RA.
        params_target_curr  = params_target.get('RA', dict())
        self.params_target += [np.array([float(params_target_curr.get('EB', 0.05))])]
        # EA_RV.
        params_target_curr  = params_target.get('RV', dict())
        self.params_target += [np.array([float(params_target_curr.get('EB', 0.55))])]
        # R_AR_SYS.
        params_target_curr  = params_target.get('SYS', dict())
        self.params_target += [np.array([float(params_target_curr.get('R_AR', 0.64))])]
        # R_VEN_SYS.
        params_target_curr  = params_target.get('SYS', dict())
        self.params_target += [np.array([float(params_target_curr.get('R_VEN', 0.32))])]        
        # V_tot_heart.
        self.params_target += [np.array([float(params_target.get('V_tot_heart', 416.75599234489175))])]

        # Used to generate noise on the observation(s).
        self.noise_trace = noise_trace

        # Discrete L^2 relative errors.
        self.l2_errors = []

    def loss(self, params):
        self.E_LA, self.E_RA, self.E_RV, \
        self.V_LA, self.V_LV, self.V_RA, self.V_RV, \
        self.p_AR_SYS, self.p_VEN_SYS, self.p_AR_PUL, self.p_VEN_PUL, \
        self.Q_AR_SYS, self.Q_VEN_SYS, self.Q_AR_PUL, self.Q_VEN_PUL, \
        self.p_LA, self.p_LV, self.p_RA, self.p_RV, \
        self.Q_MV, self.Q_AV, self.Q_TV, self.Q_PV = self.circulation_model.run(standalone      = False,
                                                                                num_cycles      = self.num_cycles,
                                                                                dt              = self.dt,
                                                                                initial_state   = self.initstate_file,
                                                                                output_folder   = self.output_folder,
                                                                                filename_output = self.filename_output,
                                                                                params          = params,
                                                                                last_HB         = True,
                                                                                save            = False)

        losses = list()
        # V_LA.
        losses.append(0.0 if self.weight_V_LA < 1e-15 else self.weight_V_LA * jnp.mean(jnp.square(self.V_LA - self.V_LA_opt)) / self.normalization_V_LA)
        # V_LV.
        losses.append(0.0 if self.weight_V_LV < 1e-15 else self.weight_V_LV * jnp.mean(jnp.square(self.V_LV - self.V_LV_opt)) / self.normalization_V_LV)
        # V_RA.
        losses.append(0.0 if self.weight_V_RA < 1e-15 else self.weight_V_RA * jnp.mean(jnp.square(self.V_RA - self.V_RA_opt)) / self.normalization_V_RA)
        # V_RV.
        losses.append(0.0 if self.weight_V_RV < 1e-15 else self.weight_V_RV * jnp.mean(jnp.square(self.V_RV - self.V_RV_opt)) / self.normalization_V_RV)
        # p_AR_SYS.
        losses.append(0.0 if self.weight_p_AR_SYS < 1e-15 else self.weight_p_AR_SYS * jnp.mean(jnp.square(self.p_AR_SYS - self.p_AR_SYS_opt)) / self.normalization_p_AR_SYS)        

        return sum(losses)        

    def func_grad(self, params_1d):
        t_init = time.time()
        params = jax_utils.reshape(params_1d, self.shape)
        l = float(np.array(self.loss_jit(params)))
        G = self.grad_jit(params)
        g = np.array(jax_utils.deshape(G))
        print('loss: %1.16e (%f s)' % (l, time.time() - t_init))
        return l, g

    def func(self, params_1d):
        params = jax_utils.reshape(params_1d, self.shape)
        l = float(np.array(self.loss_jit(params)))
        print('loss: %1.16e' % l)
        return l

    def run(self, num_cycles, dt, initial_state, output_folder = 'simulations', filename_output = 'circulation_MAP'):
        # Time settings.
        self.num_cycles       = num_cycles
        self.dt               = dt
        self.times            = np.arange(0.0, (self.num_cycles * self.circulation_model.THB) + 1e-4, self.dt)
        self.nT               = self.times.shape[0]
        self.last_cycle_init  = int(self.nT / self.num_cycles)
        self.times_last_cycle = self.times[-self.last_cycle_init:]

        self.initstate_file   = initial_state        
        self.filename_output  = filename_output
        self.output_folder    = output_folder

        # Initial conditions (for target).
        self.circulation_model.initialize(initial_state = self.initstate_file, random_init = False)

        # Compute targets.
        if self.weight_V_LA >= 1e-15:
            self.V_LA_opt     = target(self.test_case, 'VLA', 'time', 3.2, self.times_last_cycle, self.noise_trace, self.output_folder)
        if self.weight_V_RA >= 1e-15:
            self.V_RA_opt     = target(self.test_case, 'VRA', 'time', 3.2, self.times_last_cycle, self.noise_trace, self.output_folder)
        if self.weight_V_LV >= 1e-15:
            self.V_LV_opt     = target(self.test_case, 'VLV', 'time', 3.2, self.times_last_cycle, self.noise_trace, self.output_folder)
        if self.weight_V_RV >= 1e-15:
            self.V_RV_opt     = target(self.test_case, 'VRV', 'time', 3.2, self.times_last_cycle, self.noise_trace, self.output_folder)
        if self.weight_p_AR_SYS >= 1e-15:
            self.p_AR_SYS_opt = target(self.test_case, 'pARSYS', 'time', 3.2, self.times_last_cycle, self.noise_trace, self.output_folder)

        # Compute normalization factors.
        if self.weight_V_LA >= 1e-15:
            self.normalization_V_LA = jnp.mean(jnp.square(self.V_LA_opt))
        if self.weight_V_RA >= 1e-15:
            self.normalization_V_RA = jnp.mean(jnp.square(self.V_RA_opt))
        if self.weight_V_LV >= 1e-15:
            self.normalization_V_LV = jnp.mean(jnp.square(self.V_LV_opt))
        if self.weight_V_RV >= 1e-15:
            self.normalization_V_RV = jnp.mean(jnp.square(self.V_RV_opt))
        if self.weight_p_AR_SYS >= 1e-15:
            self.normalization_p_AR_SYS = jnp.mean(jnp.square(self.p_AR_SYS_opt))

        # Initial conditions (for parameters estimation).
        self.circulation_model.initialize(initial_state = self.initstate_file, random_init = self.estimate_V_tot_heart)

        # Parameters initialization (for compilation).
        params_0 = []
        # EB_LA.
        EB_LA_val = jnp.array([self.circulation_model.EB_LA])
        params_0 += [EB_LA_val]
        # a_XB.
        a_XB_val  = jnp.array([self.circulation_model.a_XB])
        params_0 += [a_XB_val]
        # EB_RA.
        EB_RA_val = jnp.array([self.circulation_model.EB_RA])
        params_0 += [EB_RA_val]
        # EA_RV.
        EA_RV_val = jnp.array([self.circulation_model.EA_RV])
        params_0 += [EA_RV_val]
        # R_AR_SYS.
        R_AR_SYS_val  = jnp.array([self.circulation_model.R_AR_SYS])
        params_0     += [R_AR_SYS_val]
        # R_VEN_SYS.
        R_VEN_SYS_val  = jnp.array([self.circulation_model.R_VEN_SYS])
        params_0      += [R_VEN_SYS_val]
        # V_tot_heart.
        V_tot_heart_val  = jnp.array([self.circulation_model.V_tot_heart_init])
        params_0        += [V_tot_heart_val]
        
        # Collect for jax.
        self.params_1d_0 = jax_utils.deshape(params_0)
        self.shape       = jax_utils.get_shape(params_0)

        # Define loss and gradients + just-in-time compilation.
        self.grad     = jax.grad(self.loss)
        self.loss_jit = jax.jit(self.loss)
        self.grad_jit = jax.jit(self.grad)
        print('compilation...')
        t_init = time.time()
        l,g = self.func_grad(self.params_1d_0)
        print('compiled (%f s)' % (time.time() - t_init))

        # Parameters initialization (for optimization).
        params_0 = []
        # EB_LA.
        EB_LA_val = np.array([self.circulation_model.EB_LA])
        params_0 += [EB_LA_val]
        # a_XB.
        a_XB_val  = np.array([self.circulation_model.a_XB])
        params_0 += [a_XB_val]        
        # EB_RA.
        EB_RA_val = np.array([self.circulation_model.EB_RA])
        params_0 += [EB_RA_val]
        # EA_RV.
        EA_RV_val = np.array([self.circulation_model.EA_RV])
        params_0 += [EA_RV_val]
        # R_AR_SYS.
        R_AR_SYS_val = np.array([self.circulation_model.R_AR_SYS])
        params_0    += [R_AR_SYS_val]
        # R_VEN_SYS.
        R_VEN_SYS_val = np.array([self.circulation_model.R_VEN_SYS])
        params_0     += [R_VEN_SYS_val]        
        # V_tot_heart.
        V_tot_heart_val = np.array([self.circulation_model.V_tot_heart_init])
        params_0       += [V_tot_heart_val]

        # Collect for jax.
        self.params_1d_0 = jax_utils.deshape(params_0)

        # Parameters bounds.
        self.define_bounds()

        # Callback of the optimizer (print parameters values + plots). 
        self.iteration    = 0
        self.refresh_rate = 0 # 0 to disable refresh.
        def callback(params_1d):
            print('====================================== iteration %d' % self.iteration)

            if self.estimate_EB_LA:
                print('EB_LA: %1.16f' % params_1d[0])
            if self.estimate_a_XB_LV:
                print('a_XB: %1.16f' % params_1d[1])
            if self.estimate_EB_RA:                
                print('EB_RA: %1.16f' % params_1d[2])
            if self.estimate_EA_RV:
                print('EA_RV: %1.16f' % params_1d[3])
            if self.estimate_R_AR_SYS:
                print('R_AR_SYS: %1.16f' % params_1d[4])
            if self.estimate_R_VEN_SYS:
                print('R_VEN_SYS: %1.16f' % params_1d[5])
            if self.estimate_V_tot_heart:                
                print('V_tot_heart: %1.16f' % params_1d[6])

            self.iteration += 1

            params = jax_utils.reshape(params_1d, self.shape)
            self.compute_errors(params)
            print('**************************************')
            print('L2 error: %1.16f' % self.l2_errors[-1])

            if self.refresh_rate > 0 and self.iteration % self.refresh_rate == 0:
                self.graph_output            = self.output_folder + '/optimization_%d.png' % self.iteration
                self.graph_output_pv         = self.output_folder + '/optimization_pv_loop_%d.png' % self.iteration
                self.graph_output_elastances = self.output_folder + '/optimization_elastances.png'

                self.E_LA, self.E_RA, self.E_RV, \
                self.V_LA, self.V_LV, self.V_RA, self.V_RV, \
                self.p_AR_SYS, self.p_VEN_SYS, self.p_AR_PUL, self.p_VEN_PUL, \
                self.Q_AR_SYS, self.Q_VEN_SYS, self.Q_AR_PUL, self.Q_VEN_PUL, \
                self.p_LA, self.p_LV, self.p_RA, self.p_RV, \
                self.Q_MV, self.Q_AV, self.Q_TV, self.Q_PV = self.circulation_model.run(standalone      = False,
                                                                                        num_cycles      = self.num_cycles,
                                                                                        dt              = self.dt,
                                                                                        initial_state   = self.initstate_file,
                                                                                        output_folder   = self.output_folder,
                                                                                        filename_output = self.filename_output,
                                                                                        params          = params,
                                                                                        last_HB         = True,
                                                                                        save            = False)
                plot_history_MAP(self)
                plt.savefig(self.graph_output)
                plot_PV_loops_MAP(self)
                plt.savefig(self.graph_output_pv)
                plot_elastances_MAP(self)
                plt.savefig(self.graph_output_elastances)
                
            return True

        # Optimization with L-BFGS.
        self.max_iter = 1000
        t_init = time.time()
        ret = spopt.minimize(fun = self.func_grad, x0 = self.params_1d_0, jac = True,
                             method = 'L-BFGS-B', bounds = self.bounds,
                             options = {'ftol': 1e-100, 'gtol': 1e-100, 'maxiter': self.max_iter},
                             callback = callback)
        opt_time = time.time() - t_init
        print('Optimization time: %f s' % opt_time)

        # Output results.
        print(ret['message'])
        print('nit: %d' % ret['nit'])
        print('Parameters (estimated):')
        print(ret['x'])
        print('Parameters (target):')
        params_target = np.array(flatten_list_array([arr.tolist() for arr in self.params_target]))
        print(params_target)
        print('Discrete L^2 relative error:')
        print(self.l2_errors[-1])

        # Save discrete L^2 relative errors.
        dict_data = {'it'         : list(range(1, self.iteration + 1)),
                     'l2_error'   : self.l2_errors}
        pd.DataFrame(dict_data).to_csv(self.output_folder + '/' + self.filename_output + '_errors.csv', index = False)

        # Final run and save results.
        params_1d_opt = ret['x']
        loss_opt, grad_opt = self.func_grad(params_1d_opt)
        print('Loss (target):')
        print(loss_opt)
        print('Gradient l2 norm (target):')
        print(LA.norm(grad_opt))

        self.params_opt = jax_utils.reshape(params_1d_opt, self.shape)
        self.circulation_model.run(standalone      = False,
                                   num_cycles      = self.num_cycles,
                                   dt              = self.dt,
                                   initial_state   = self.initstate_file,
                                   output_folder   = self.output_folder,
                                   filename_output = self.filename_output,
                                   params          = self.params_opt,
                                   last_HB         = True,
                                   save            = True)

        return opt_time, self.l2_errors[-1], ret['nit'], loss_opt

    def define_bounds(self):
        self.n_est_params = 0
        self.bounds       = []

        if self.estimate_EB_LA:
            self.bounds += [[0.15 - 0.5 * 0.15, 0.15 + 0.5 * 0.15]]
            self.n_est_params += 1
        else:
            self.bounds += [[self.circulation_model.EB_LA, self.circulation_model.EB_LA]]

        if self.estimate_a_XB_LV:
            self.bounds += [[80.0, 320.0]]
            self.n_est_params += 1
        else:
            self.bounds += [[self.circulation_model.a_XB, self.circulation_model.a_XB]]

        if self.estimate_EB_RA:
            self.bounds += [[0.05 - 0.5 * 0.05, 0.05 + 0.5 * 0.05]]
            self.n_est_params += 1
        else:
            self.bounds += [[self.circulation_model.EB_RA, self.circulation_model.EB_RA]]

        if self.estimate_EA_RV:
            self.bounds += [[0.55 - 0.5 * 0.55, 0.55 + 0.5 * 0.55]]
            self.n_est_params += 1
        else:
            self.bounds += [[self.circulation_model.EA_RV, self.circulation_model.EA_RV]]

        if self.estimate_R_AR_SYS:
            self.bounds += [[0.54, 1.2]]
            self.n_est_params += 1
        else:
            self.bounds += [[self.circulation_model.R_AR_SYS, self.circulation_model.R_AR_SYS]]

        if self.estimate_R_VEN_SYS:
            self.bounds += [[0.18, 0.40]]
            self.n_est_params += 1
        else:
            self.bounds += [[self.circulation_model.R_VEN_SYS, self.circulation_model.R_VEN_SYS]]

        if self.estimate_V_tot_heart:
            self.bounds += [[200.0, 600.0]]
            self.n_est_params += 1
        else:
            self.bounds += [[self.circulation_model.V_tot_heart_init, self.circulation_model.V_tot_heart_init]]

    def compute_errors(self, params):
        params_target = np.array(flatten_list_array([arr.tolist() for arr in self.params_target]))
        params        = np.array(flatten_list_array([arr.tolist() for arr in params]))

        l2_error = 0.
        for t, o in zip(params_target, params):
            l2_error += ((t - o) / t)**2
        l2_error = l2_error / self.n_est_params
        l2_error = math.sqrt(l2_error)
        self.l2_errors.append(l2_error)