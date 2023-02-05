import numpy as np
import itertools
import jax.numpy as jnp

def get_shape(x):
    index = 0
    def get_shape_index(x):
        nonlocal index
        if isinstance(x, list) or isinstance(x, tuple):
            return [get_shape_index(xi) for xi in x]
        else:
            ret = {'index': index, 'shape': x.shape}
            index += np.prod(x.shape)
            return ret
    return get_shape_index(x)

def deshape(X):
    return jnp.concatenate([x.reshape(-1) for x in itertools.chain(*X)])

def reshape(x, shape):
    if isinstance(shape, list):
        return [reshape(x, shape_i) for shape_i in shape]
    else:
        return x[shape['index']:shape['index']+np.prod(shape['shape'])].reshape(shape['shape'])

def piecewise_const(t, time_points, values):
    return jnp.sum( \
               jnp.heaviside(t - time_points[:-1], 1.0) * \
               jnp.heaviside(time_points[1:]  - t, 0.0) * \
               values)