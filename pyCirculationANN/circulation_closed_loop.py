import numpy as np
import pandas as pd
import json
import csv
import time
import random

from scipy.optimize import root

import jax
import jax_utils
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.config import config
config.update("jax_enable_x64", True)

from utils import *

# For reproducibility among multiple runs.
random.seed(0)

class circulation_closed_loop:

    def __init__(self, test_case, options, params):

        if isinstance(options, str):
            with open(options, mode='r', newline='') as inputfile:
                options = json.loads(inputfile.read())

        # Heartbeat.
        self.BPM = float(options.get('BPM', 72)) # [1 / min].
        # Heartbeat period
        self.THB = 60. / self.BPM # [s].

        # Left atrium.
        options_curr = options.get('LA', dict())
        # Active elastance.
        self.EA_LA   = float(options_curr.get('EA', 0.07))                         # [mmHg / ml].
        # Passive elastance.
        if (test_case == 'all' or test_case == 'atria'):
            self.EB_LA = float(random.uniform(0.15 - 0.5 * 0.15, 0.15 + 0.5 * 0.15)) # [mmHg / ml].
        else:
            self.EB_LA = float(options_curr.get('EB', 0.09))                         # [mmHg / ml].
        # Time of contraction.
        self.TC_LA   = float(options_curr.get('TC', 0.17)) * self.THB              # [s].
        # Time of relaxation.
        self.TR_LA   = float(options_curr.get('TR', 0.17)) * self.THB              # [s].
        # Initial time of contraction.
        self.tC_LA   = float(options_curr.get('tC', 0.80)) * self.THB              # [s].
        # Reference volume.
        self.V0_LA   = float(options_curr.get('V0', 4.0))                          # [ml].
        # Elastance over time.
        self.E_LA    = self.time_varying_elastance(self.EA_LA, self.EB_LA, self.tC_LA, self.TC_LA, self.TR_LA)

        # Left ventricle.
        options_curr = options.get('LV', dict())
        # Cardiomyocytes contractility.
        if (test_case == 'none' or test_case == 'atria'):
            self.a_XB = float(options_curr.get('a_XB', 160.0)) # [MPa].
        else:
            self.a_XB = float(random.uniform(80.0, 320.0))     # [MPa].

        # Right atrium.
        options_curr = options.get('RA', dict())
        # Active elastance.
        self.EA_RA   = float(options_curr.get('EA', 0.06))                         # [mmHg / ml].
        # Passive elastance.
        if (test_case == 'all' or test_case == 'atria'):
            self.EB_RA = float(random.uniform(0.05 - 0.5 * 0.05, 0.05 + 0.5 * 0.05)) # [mmHg / ml].
        else:
            self.EB_RA = float(options_curr.get('EB', 0.07))                         # [mmHg / ml].
        # Time of contraction.
        self.TC_RA   = float(options_curr.get('TC', 0.17)) * self.THB              # [s].
        # Time of relaxation.
        self.TR_RA   = float(options_curr.get('TR', 0.17)) * self.THB              # [s].
        # Initial time of contraction.
        self.tC_RA   = float(options_curr.get('tC', 0.80)) * self.THB              # [s].
        # Reference volume.
        self.V0_RA   = float(options_curr.get('V0', 4.0))                          # [ml].
        # Elastance over time.
        self.E_RA    = self.time_varying_elastance(self.EA_RA, self.EB_RA, self.tC_RA, self.TC_RA, self.TR_RA)

        # Right ventricle.
        options_curr = options.get('RV', dict())
        # Active elastance.
        if (test_case == 'all'):
            self.EA_RV = float(random.uniform(0.55 - 0.5 * 0.55, 0.55 + 0.5 * 0.55)) # [mmHg / ml].
        else:
            self.EA_RV = float(options_curr.get('EA', 0.55))                         # [mmHg / ml].
        # Passive elastance.
        self.EB_RV   = float(options_curr.get('EB', 0.05))                         # [mmHg / ml].
        # Time of contraction.
        self.TC_RV   = float(options_curr.get('TC', 0.34)) * self.THB              # [s].
        # Time of relaxation.
        self.TR_RV   = float(options_curr.get('TR', 0.17)) * self.THB              # [s].
        # Initial time of contraction.
        self.tC_RV   = float(options_curr.get('tC', 0.00)) * self.THB              # [s].
        # Initial time of contraction.
        self.V0_RV   = float(options_curr.get('V0', 10.0))                         # [ml].
        # Elastance over time.
        self.E_RV    = self.time_varying_elastance(self.EA_RV, self.EB_RV, self.tC_RV, self.TC_RV, self.TR_RV)

        # Valves.
        heavisideMY  = lambda x: jnp.arctan(jnp.pi / 2 * x * 200) * 1 / jnp.pi + 0.5
        options_curr = options.get('valves', dict())
        # Minimum resistance.
        Rmin         = float(options_curr.get('Rmin', 0.0075))  # [mmHg s / ml].
        # Maximum resistance.
        Rmax         = float(options_curr.get('Rmax', 75006.2)) # [mmHg s / ml].
        # Mitral valve.
        self.R_MV    = lambda w, v: 10.**(jnp.log10(Rmin) + (jnp.log10(Rmax) - jnp.log10(Rmin)) * heavisideMY(v-w))
        # Aortic valve.
        self.R_AV    = lambda w, v: 10.**(jnp.log10(Rmin) + (jnp.log10(Rmax) - jnp.log10(Rmin)) * heavisideMY(v-w))
        # Tricuspid valve.
        self.R_TV    = lambda w, v: 10.**(jnp.log10(Rmin) + (jnp.log10(Rmax) - jnp.log10(Rmin)) * heavisideMY(v-w))
        # Pulmonary valve.
        self.R_PV    = lambda w, v: 10.**(jnp.log10(Rmin) + (jnp.log10(Rmax) - jnp.log10(Rmin)) * heavisideMY(v-w))

        # Systemic circulation.
        options_curr   = options.get('SYS', dict())
        # Arterial systemic resistance.
        if (test_case == 'none' or test_case == 'atria'):
            self.R_AR_SYS = float(options_curr.get('R_AR', 0.8))  # [mmHg s / ml].
        else:
            self.R_AR_SYS = float(random.uniform(0.54, 1.2))      # [mmHg s / ml].
        # Arterial systemic capacitance.
        self.C_AR_SYS  = float(options_curr.get('C_AR', 1.2))     # [ml / mmHg].
        # Venous systemic resistance.
        if (test_case == 'all'):
            self.R_VEN_SYS = float(random.uniform(0.18, 0.40))      # [mmHg s / ml].
        else:
            self.R_VEN_SYS = float(options_curr.get('R_VEN', 0.26)) # [mmHg s / ml].
        # Venous systemic capacitance.
        self.C_VEN_SYS = float(options_curr.get('C_VEN', 60.))    # [ml / mmHg].
        # Arterial systemic inductance.
        self.L_AR_SYS  = float(options_curr.get('L_AR', 5e-3))    # [mmHg s^2 / ml].
        # Venous systemic inductance.
        self.L_VEN_SYS = float(options_curr.get('L_VEN', 5e-4))   # [mmHg s^2 / ml].

        # Pulmonary circulation.
        options_curr   = options.get('PUL', dict())
        # Arterial pulmonary resistance.
        self.R_AR_PUL  = float(options_curr.get('R_AR', 0.1625))  # [mmHg s / ml].
        # Arterial pulmonary capacitance.
        self.C_AR_PUL  = float(options_curr.get('C_AR', 10.))     # [ml / mmHg].
        # Venous pulmonary resistance.
        self.R_VEN_PUL = float(options_curr.get('R_VEN', 0.1625)) # [mmHg s / ml].
        # Venous pulmonary capacitance.
        self.C_VEN_PUL = float(options_curr.get('C_VEN', 16.))    # [ml / mmHg].
        # Arterial pulmonary inductance.
        self.L_AR_PUL  = float(options_curr.get('L_AR', 5e-4))    # [mmHg s^2 / ml].
        # Venous pulmonary inductance.
        self.L_VEN_PUL = float(options_curr.get('L_VEN', 5e-4))   # [mmHg s^2 / ml].

        self.flux_through_valve = lambda p1, p2, R: (p1 - p2) / R(p1, p2)

        # For V_LV^0D = V_LV^ANN.
        self.epsilon = 1e-5
        # Right hand side.
        if params is False:
            self.rhs = lambda state, t, params: self.circulation_model(t,
                                                                       state,
                                                                       self.E_LA(t),
                                                                       self.a_XB,
                                                                       self.E_RA(t),
                                                                       self.E_RV(t),
                                                                       self.R_AR_SYS,
                                                                       self.R_VEN_SYS)
        else:
            self.rhs = lambda state, t, params: self.circulation_model(t,
                                                                       state,
                                                                       self.time_varying_elastance(self.EA_LA, params[0][0], self.tC_LA, self.TC_LA, self.TR_LA)(t),
                                                                       params[1][0],
                                                                       self.time_varying_elastance(self.EA_RA, params[2][0], self.tC_RA, self.TC_RA, self.TR_RA)(t),
                                                                       self.time_varying_elastance(params[3][0], self.EB_RV, self.tC_RV, self.TC_RV, self.TR_RV)(t),
                                                                       params[4][0],
                                                                       params[5][0])

        # Import ANN.
        self.ANN = ANNmodel('data/ANN-based-ROM.mat')
        print('LV ANN has been imported successfully!')

    def time_varying_elastance(self, EA, EB, time_C, duration_C, duration_R):
        time_R = time_C + duration_C
        e = lambda t: 0.5 * (1 - jnp.cos(jnp.pi / duration_C * (jnp.mod(t - time_C, self.THB)))) * (0 <= jnp.mod(t - time_C, self.THB)) * (jnp.mod(t - time_C, self.THB) < duration_C) + \
                      0.5 * (1 + jnp.cos(jnp.pi / duration_R * (jnp.mod(t - time_R, self.THB)))) * (0 <= jnp.mod(t - time_R, self.THB)) * (jnp.mod(t - time_R, self.THB) < duration_R)
        return lambda t: EA * jnp.clip(e(t), 0.0, 1.0) + EB

    def initialize(self, initial_state = dict(), random_init = False):
        if isinstance(initial_state, str):
            with open(initial_state, mode='r', newline='') as inputfile:
                initial_state = json.loads(inputfile.read())

        if random_init:
            self.V_LA_init = float(random.uniform(50., 100.))  # [ml].
            self.V_RA_init = float(random.uniform(50., 100.))  # [ml].
            # For consistency with the ANN.
            self.V_LV_init = float(initial_state.get('V_LV', 120.)) # [ml]. 
            self.V_RV_init = float(random.uniform(100., 200.))      # [ml].
        else:
            self.V_LA_init = float(initial_state.get('V_LA',  65.)) # [ml].
            self.V_RA_init = float(initial_state.get('V_RA',  65.)) # [ml].
            self.V_LV_init = float(initial_state.get('V_LV', 120.)) # [ml].
            self.V_RV_init = float(initial_state.get('V_RV', 145.)) # [ml].

        self.p_AR_SYS_init  = float(initial_state.get('p_AR_SYS' ,  80.)) # [mmHg].
        self.p_VEN_SYS_init = float(initial_state.get('p_VEN_SYS',  30.)) # [mmHg].
        self.p_AR_PUL_init  = float(initial_state.get('p_AR_PUL' ,  35.)) # [mmHg].
        self.p_VEN_PUL_init = float(initial_state.get('p_VEN_PUL',  24.)) # [mmHg].

        self.Q_AR_SYS_init  = float(initial_state.get('Q_AR_SYS' ,   0.)) # [ml / s].
        self.Q_VEN_SYS_init = float(initial_state.get('Q_VEN_SYS',   0.)) # [ml / s].
        self.Q_AR_PUL_init  = float(initial_state.get('Q_AR_PUL' ,   0.)) # [ml / s].
        self.Q_VEN_PUL_init = float(initial_state.get('Q_VEN_PUL',   0.)) # [ml / s].

        self.V_LV_ANN_init = self.ANN.initial_state[0] # [ml].
        if (self.ANN.initial_state.size > 1):
            self.mute_var = self.ANN.initial_state[1:]
        else:
            self.mute_var = []
        if (np.abs(self.V_LV_ANN_init - self.V_LV_init) > 1e-3):
            raise Exception('V_LV_ANN and V_LV must be the same!')
        self.p_LV_init = 0.2 * (self.V_LV_init - 42.0)
        res = root(self.fun_init_state, self.p_LV_init, method = 'lm')
        self.p_LV_init = res.x[0]

        self.V_tot_heart_init = self.V_LA_init + self.V_LV_init + self.V_RA_init + self.V_RV_init
        self.V_tot_SYS_init   = self.C_AR_SYS  * self.p_AR_SYS_init \
                              + self.C_VEN_SYS * self.p_VEN_SYS_init
        self.V_tot_PUL_init   = self.C_AR_PUL  * self.p_AR_PUL_init  \
                              + self.C_VEN_PUL * self.p_VEN_PUL_init
        self.V_tot_init       = self.V_tot_heart_init + self.V_tot_SYS_init + self.V_tot_PUL_init

    # V_LA          = state[0].
    # V_LV^0D       = state[1].
    # V_RA          = state[2].
    # V_RV          = state[3].
    # p_AR_SYS      = state[4].
    # p_VEN_SYS     = state[5].
    # p_AR_PUL      = state[6].
    # p_VEN_PUL     = state[7].
    # Q_AR_SYS      = state[8].
    # Q_VEN_SYS     = state[9].
    # Q_AR_PUL      = state[10].
    # Q_VEN_PUL     = state[11].
    # p_LV          = state[12].
    # (V_LV^ANN, z) = state[13, :].
    # self.p_LA     = ELA * (state[0] - self.V0_LA).
    # self.p_RA     = ERA * (state[2] - self.V0_RA).
    # self.p_RV     = ERV * (state[3] - self.V0_RV).
    # self.Q_MV     = self.flux_through_valve(ELA * (state[0] - self.V0_LA), state[12], self.R_MV).
    # self.Q_AV     = self.flux_through_valve(state[12], state[4], self.R_AV).
    # self.Q_TV     = self.flux_through_valve(ERA * (state[2] - self.V0_RA), ERV * (state[3] - self.V0_RV), self.R_TV).
    # self.Q_PV     = self.flux_through_valve(ERV * (state[3] - self.V0_RV), state[6], self.R_PV).
    def circulation_model(self, t, state, ELA, aXB, ERA, ERV, RARSYS, RVENSYS):
        u = [jnp.cos(2 * jnp.pi * t / self.THB), jnp.sin(2 * np.pi * t / self.THB), state[12], aXB]
        u = jnp.array(u)

        z = []
        for index in range(13, 13 + self.ANN.initial_state.size):
            z = z + [state[index]]
        ANN_rhs = self.ANN.rhs(jnp.array(z), u)

        out = [# Q_VEN_PUL - Q_MV.
               state[11] - self.flux_through_valve(ELA * (state[0] - self.V0_LA), state[12], self.R_MV),
               # Q_MV - Q_AV.
               self.flux_through_valve(ELA * (state[0] - self.V0_LA), state[12], self.R_MV) - self.flux_through_valve(state[12], state[4], self.R_AV),
               # Q_VEN_SYS - Q_TV.
               state[9] - self.flux_through_valve(ERA * (state[2] - self.V0_RA), ERV * (state[3] - self.V0_RV), self.R_TV),
               # Q_TV - Q_PV.
               self.flux_through_valve(ERA * (state[2] - self.V0_RA), ERV * (state[3] - self.V0_RV), self.R_TV) - self.flux_through_valve(ERV * (state[3] - self.V0_RV), state[6], self.R_PV),
               # (Q_AV - Q_AR_SYS) / C_AR_SYS.
               (self.flux_through_valve(state[12], state[4], self.R_AV) - state[8]) / self.C_AR_SYS,
               # (Q_AR_SYS - Q_VEN_SYS) / C_VEN_SYS.
               (state[8] - state[9]) / self.C_VEN_SYS,
               # (Q_PV - Q_AR_PUL) / C_AR_PUL.
               (self.flux_through_valve(ERV * (state[3] - self.V0_RV), state[6], self.R_PV) - state[10]) / self.C_AR_PUL,
               # (Q_AR_PUL - Q_VEN_PUL) / C_VEN_PUL.
               (state[10] - state[11]) / self.C_VEN_PUL,
               # - (R_AR_SYS * Q_AR_SYS + p_VEN_SYS - p_AR_SYS) / L_AR_SYS.
               - (RARSYS * state[8] + state[5] - state[4]) / self.L_AR_SYS,
               # - (R_VEN_SYS * Q_VEN_SYS + p_RA - p_VEN_SYS) / L_VEN_SYS.
               - (RVENSYS * state[9] + ERA*(state[2] - self.V0_RA) - state[5]) / self.L_VEN_SYS,
               # - (R_AR_PUL * Q_AR_PUL + p_VEN_PUL - p_AR_PUL) / L_AR_PUL.
               - (self.R_AR_PUL  * state[10]  + state[7] - state[6]) / self.L_AR_PUL,
               # - (R_VEN_PUL * Q_VEN_PUL + p_LA - p_VEN_PUL) / L_VEN_PUL.
               - (self.R_VEN_PUL * state[11] + ELA * (state[0] - self.V0_LA) - state[7]) / self.L_VEN_PUL,
               # V_LV^0D - V_LV^ANN.
               (state[1] - state[13]) / self.epsilon
               ]

        # ANN(z, cos(2 * pi * t / THB), sin(2 * pi * t / THB), p_LV, a_XB).
        for index in range(self.ANN.initial_state.size):
            out = out + [ANN_rhs[index]]

        return jnp.stack(out)

    def fun_init_state(self, pLV):
        N = self.ANN.initial_state.size
        if type(pLV) is np.ndarray:
            pLV = pLV[0]
        u = [1.0, 0.0, pLV, self.a_XB]
        u = jnp.array(u)

        dx = self.ANN.rhs(self.ANN.initial_state, u)

        if N == 1:
            return dx
        else:
            return np.linalg.norm(dx)

    def forward_model(self, times, index_init, times_eval, params = None):
        # Initial conditions.
        if params is None:
            state_0 = jnp.array(flatten_list_array([self.V_LA_init,
                                                    self.V_LV_init,
                                                    self.V_RA_init,
                                                    self.V_RV_init,
                                                    self.p_AR_SYS_init,
                                                    self.p_VEN_SYS_init,
                                                    self.p_AR_PUL_init,
                                                    self.p_VEN_PUL_init,
                                                    self.Q_AR_SYS_init,
                                                    self.Q_VEN_SYS_init,
                                                    self.Q_AR_PUL_init,
                                                    self.Q_VEN_PUL_init,
                                                    self.p_LV_init,
                                                    self.V_LV_ANN_init,
                                                    self.mute_var]))           
        else:
            state_0 = jnp.array(flatten_list_array([self.V_LA_init * (params[6][0] - self.V_LV_init) / (self.V_tot_heart_init - self.V_LV_init),
                                                    self.V_LV_init,
                                                    self.V_RA_init * (params[6][0] - self.V_LV_init) / (self.V_tot_heart_init - self.V_LV_init),
                                                    self.V_RV_init * (params[6][0] - self.V_LV_init) / (self.V_tot_heart_init - self.V_LV_init),
                                                    self.p_AR_SYS_init,
                                                    self.p_VEN_SYS_init,
                                                    self.p_AR_PUL_init,
                                                    self.p_VEN_PUL_init,
                                                    self.Q_AR_SYS_init,
                                                    self.Q_VEN_SYS_init,
                                                    self.Q_AR_PUL_init,
                                                    self.Q_VEN_PUL_init,
                                                    self.p_LV_init,
                                                    self.V_LV_ANN_init,
                                                    self.mute_var]))

        # Time integration.
        y = odeint(self.rhs, state_0, times, params)

        # Get elastances.
        if params is not None:
            self.E_LA = self.time_varying_elastance(self.EA_LA, params[0][0], self.tC_LA, self.TC_LA, self.TR_LA)
            self.E_RA = self.time_varying_elastance(self.EA_RA, params[2][0], self.tC_RA, self.TC_RA, self.TR_RA)
            self.E_RV = self.time_varying_elastance(params[3][0], self.EB_RV, self.tC_RV, self.TC_RV, self.TR_RV)

        # Get outputs.
        self.V_LA      = y[-index_init:, 0]
        self.V_LV      = y[-index_init:, 1]
        self.V_RA      = y[-index_init:, 2]
        self.V_RV      = y[-index_init:, 3]
        self.p_AR_SYS  = y[-index_init:, 4]
        self.p_VEN_SYS = y[-index_init:, 5]
        self.p_AR_PUL  = y[-index_init:, 6]
        self.p_VEN_PUL = y[-index_init:, 7]
        self.Q_AR_SYS  = y[-index_init:, 8]
        self.Q_VEN_SYS = y[-index_init:, 9]
        self.Q_AR_PUL  = y[-index_init:, 10]
        self.Q_VEN_PUL = y[-index_init:, 11]
        self.p_LA      = self.E_LA(times_eval) * (y[-index_init:, 0] - self.V0_LA)
        self.p_LV      = y[-index_init:, 12]
        self.p_RA      = self.E_RA(times_eval) * (y[-index_init:, 2] - self.V0_RA)
        self.p_RV      = self.E_RV(times_eval) * (y[-index_init:, 3] - self.V0_RV)
        self.Q_MV      = self.flux_through_valve(self.E_LA(times_eval) * (y[-index_init:, 0] - self.V0_LA), self.p_LV, self.R_MV)
        self.Q_AV      = self.flux_through_valve(self.p_LV, y[-index_init:, 4], self.R_AV)
        self.Q_TV      = self.flux_through_valve(self.E_RA(times_eval) * (y[-index_init:, 2] - self.V0_RA), \
                                                 self.E_RV(times_eval) * (y[-index_init:, 3] - self.V0_RV), self.R_TV)
        self.Q_PV      = self.flux_through_valve(self.E_RV(times_eval) * (y[-index_init:, 3] - self.V0_RV), y[-index_init:, 6], self.R_PV)
        self.V_LV_ANN  = y[-index_init:, 13]

    def run(self, standalone = True, T = None, num_cycles = None, dt = 1e-2, initial_state = None, output_folder = 'simulations', filename_output = 'circulation', params = None, last_HB = True, save = True):
        # Time settings.
        if (T is None and num_cycles is None) or (T is not None and num_cycles is not None):
            raise Exception('One among T and num_cycles should be not None.')
        if num_cycles is not None:
            T = self.THB * num_cycles
        times            = np.arange(0.0, T + 1e-4, dt)
        nT               = times.shape[0]
        last_cycle_init  = int(nT / num_cycles)
        times_last_cycle = times[-last_cycle_init:]

        # Initial conditions.
        if (standalone):
            self.initialize(initial_state = initial_state)

        if last_HB:
            # Forward model.
            self.forward_model(times, last_cycle_init, times_last_cycle, params)
            
            if save:
                # Save output.
                self.save(last_cycle_init, times_last_cycle, output_folder, filename_output, params)
        else:
            # Forward model.
            self.forward_model(times, 0, times, params)

            if save:
                # Save output.
                self.save(0, times, output_folder, filename_output, params)

        return self.E_LA, self.E_RA, self.E_RV, self.V_LA, self.V_LV, self.V_RA, self.V_RV, self.p_AR_SYS, self.p_VEN_SYS, self.p_AR_PUL, self.p_VEN_PUL, self.Q_AR_SYS, self.Q_VEN_SYS, self.Q_AR_PUL, self.Q_VEN_PUL, self.p_LA, self.p_LV, self.p_RA, self.p_RV, self.Q_MV, self.Q_AV, self.Q_TV, self.Q_PV
    
    def save(self, index_init, times_eval, output_folder, filename_output, params = None):   
        # Output of the numerical simulation.
        dict_data = {'t'         : times_eval,
                     'V_LA'      : self.V_LA,
                     'V_LV'      : self.V_LV_ANN,
                     'V_RA'      : self.V_RA,
                     'V_RV'      : self.V_RV,
                     'p_AR_SYS'  : self.p_AR_SYS,
                     'p_VEN_SYS' : self.p_VEN_SYS,
                     'p_AR_PUL'  : self.p_AR_PUL,
                     'p_VEN_PUL' : self.p_VEN_PUL,
                     'Q_AR_SYS'  : self.Q_AR_SYS,
                     'Q_VEN_SYS' : self.Q_VEN_SYS,
                     'Q_AR_PUL'  : self.Q_AR_PUL,
                     'Q_VEN_PUL' : self.Q_VEN_PUL,
                     'p_LA'      : self.p_LA,
                     'p_LV'      : self.p_LV,
                     'p_RA'      : self.p_RA,
                     'p_RV'      : self.p_RV,
                     'Q_MV'      : self.Q_MV,
                     'Q_AV'      : self.Q_AV,
                     'Q_TV'      : self.Q_TV,
                     'Q_PV'      : self.Q_PV}

        if params is None:
            dict_data['E_LA'] = self.E_LA(times_eval)
            dict_data['E_RA'] = self.E_RA(times_eval)
            dict_data['E_RV'] = self.E_RV(times_eval)
        else:
            dict_data['E_LA'] = self.time_varying_elastance(self.EA_LA, params[0][0], self.tC_LA, self.TC_LA, self.TR_LA)(times_eval)
            dict_data['E_RA'] = self.time_varying_elastance(self.EA_RA, params[2][0], self.tC_RA, self.TC_RA, self.TR_RA)(times_eval)
            dict_data['E_RV'] = self.time_varying_elastance(params[3][0], self.EB_RV, self.tC_RV, self.TC_RV, self.TR_RV)(times_eval)

        pd.DataFrame(dict_data).to_csv(output_folder + '/' + filename_output + '_data.csv', index = False)

        # Parameters.
        if params is not None:
            dict_params                 = {}
            dict_params['EB_LA']        = params[0]
            dict_params['a_XB']         = params[1]
            dict_params['EB_RA']        = params[2]
            dict_params['EA_RV']        = params[3]        
            dict_params['R_AR_SYS']     = params[4]
            dict_params['R_VEN_SYS']    = params[5]
            dict_params['V_tot_heart']  = params[6]

            pd.DataFrame(dict_params).to_csv(output_folder + '/' + filename_output + '_params.csv', index = False)
        
        print('Output saved!')