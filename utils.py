import os
import json
import csv
import time
import numpy as np
import pandas as pd
import configparser

import scipy.io as sio
from scipy import interpolate

import jax
import jax_utils
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

class ANNmodel:
    def __init__(self, path):
        data = sio.loadmat(path)

        self.initial_state = data['x0'][:, 0]

        self.f_weights = data['W'][0]
        self.f_biases  = data['T'][0]

        self.rhs = lambda x, u: self.ANN(jnp.concatenate([u, x]), self.f_weights, self.f_biases)


    def ANN(self, input, weights, biases):
        y = input
        for i in range(len(weights)):
            y = jnp.matmul(weights[i], y) - biases[i][:,0]
            if i < len(weights) - 1:
                y = jnp.tanh(y)
        return y

def target(test_case, target_label, time_label, t_min, times, noise_trace, output_folder):
    if (test_case == 'LV' or test_case == 'all'):
        history = pd.read_csv('data/FOM_LV_all.csv', delimiter=',').apply(pd.to_numeric, errors='coerce')
    else:
        history = pd.read_csv('data/FOM_atria.csv', delimiter=',').apply(pd.to_numeric, errors='coerce')
    history    = history[history[time_label] >= t_min]
    tck        = interpolate.splrep([x for x in history[time_label]], history[target_label])
    target     = interpolate.splev(times, tck)
    target_vec = np.asarray(target)
    np.savetxt(output_folder + '/' + target_label + '_target.csv', target_vec, delimiter=",")

    target = target + noise_trace

    return target

def flatten_list_array(_list_array):
    flat_list = []
    for element in _list_array:
        if isinstance(element, list) or isinstance(element, np.ndarray):
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list