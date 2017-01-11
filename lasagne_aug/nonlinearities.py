# coding:utf-8
"""
Non-linear activation functions for artificial neurons.
"""

import theano.tensor as tensor

def softmax(x):
    """
    softmax activation for 2D or 3D tensor
    :param x:
    :return:
    """
    ndim = x.ndim
    if ndim == 2:
        return tensor.nnet.softmax(x)
    elif ndim == 3:
        e = tensor.exp(x - tensor.max(x, axis=-1, keepdims=True))
        s = tensor.sum(e, axis=-1, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor '
                         'that is not 2D or 3D. '
                         'Here, ndim=' + str(ndim))
