# coding:utf-8
import theano.tensor as tensor


def softmax(x):
    ndim = x.ndim
    if ndim == 2:
        return tensor.nnet.softmax(x)
    elif ndim == 3:
        e = tensor.exp(x - tensor.max(x, axis=-1, keepdims=True))
        s = tensor.sum(e, axis=-1, keepdims=True)
        return e / s
    else:
        raise Exception('Cannot apply softmax to a tensor that is not 2D or 3D. ' +
                        'Here, ndim=' + str(ndim))


def softplus(x):
    return tensor.nnet.softplus(x)


def softsign(x):
    return tensor.nnet.nnet.softsign(x)


def relu(x, alpha=0., max_value=None):
    x = tensor.nnet.relu(x, alpha)
    if max_value is not None:
        x = tensor.minimum(x, max_value)
    return x


def tanh(x):
    return tensor.tanh(x)


def sigmoid(x):
    return tensor.nnet.sigmoid(x)


def hard_sigmoid(x):
    return tensor.nnet.hard_sigmoid(x)


def linear(x):
    '''
    The function returns the variable that is passed in, so all types work.
    '''
    return x


def get(identifier):
    if identifier is None:
        return linear
    return globals().get(identifier)
