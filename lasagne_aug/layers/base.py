# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as  tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def variable(value, dtype='float32', name=None):
    '''Instantiate a tensor variable.
    '''
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)

def random_uniform_variable(shape, low, high, dtype='float32', name=None):
    return variable(np.random.uniform(low=low, high=high, size=shape),
                    dtype=dtype, name=name)


def random_normal_variable(shape, mean, scale, dtype='float32', name=None):
    return variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                    dtype=dtype, name=name)


def _dropout(x, level, seed=None):
    if level < 0. or level >= 1:
        raise Exception('Dropout level must be in interval [0, 1[.')
    if seed is None:
        seed = np.random.randint(1, 10e6)
    rng = RandomStreams(seed=seed)
    retain_prob = 1. - level
    x *= rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
    x /= retain_prob
    return x

def _repeat(x, n):
    '''Repeat a 2D tensor.

    If x has shape (samples, dim) and n=2,
    the output will have shape (samples, 2, dim).
    '''
    assert x.ndim == 2
    x = x.dimshuffle((0, 'x', 1))
    return tensor.extra_ops.repeat(x, n, axis=1)