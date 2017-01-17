import theano.tensor as tensor, numpy as np
from .base import variable

class Regularizer(object):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {'name': self.__class__.__name__}


class EigenvalueRegularizer(Regularizer):
    '''This takes a constant that controls
    the regularization by Eigenvalue Decay on the
    current layer and outputs the regularized
    loss (evaluated on the training data) and
    the original loss (evaluated on the
    validation data).
    '''
    def __init__(self, k):
        self.k = k
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        power = 9  # number of iterations of the power method
        W = self.p
        if W.ndim > 2:
            raise Exception('Eigenvalue Decay regularizer '
                            'is only available for dense '
                            'and embedding layers.')
        WW = tensor.dot(tensor.transpose(W), W)
        dim1, dim2 = WW.shape.eval()  # number of neurons in the layer

        # power method for approximating the dominant eigenvector:
        o = variable(np.ones([dim1, 1]))  # initial values for the dominant eigenvector
        domin_eigenvect = tensor.dot(WW, o)
        for n in range(power - 1):
            domin_eigenvect = tensor.dot(WW, domin_eigenvect)

        WWd = tensor.dot(WW, domin_eigenvect)

        # the corresponding dominant eigenvalue:
        domin_eigenval = tensor.dot(tensor.transpose(WWd), domin_eigenvect) / tensor.dot(tensor.transpose(domin_eigenvect), domin_eigenvect)
        regularized_loss = loss + (domin_eigenval ** 0.5) * self.k  # multiplied by the given regularization gain

        return K.in_train_phase(regularized_loss[0, 0], loss)   # Todo: replace all in_train_phase  [DV]


class WeightRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0., dtype='float32'):
        self.l1 = np.asarray(l1, dtype=dtype)
        self.l2 = np.asarray(l2, dtype=dtype)
        self.uses_learning_phase = True

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        if not hasattr(self, 'p'):
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. '
                            'Check that you are not passing '
                            'a WeightRegularizer instead of an '
                            'ActivityRegularizer '
                            '(i.e. activity_regularizer="l2" instead '
                            'of activity_regularizer="activity_l2".')
        regularized_loss = loss + tensor.sum(tensor.abs_(self.p)) * self.l1
        regularized_loss += tensor.sum(tensor.sqr(self.p)) * self.l2
        return K.in_train_phase(regularized_loss, loss)     # Todo: replace all in_train_phase  [DV]

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}


class ActivityRegularizer(Regularizer):
    def __init__(self, l1=0., l2=0., dtype='float32'):
        self.l1 = np.asarray(l1, dtype=dtype)
        self.l2 = np.asarray(l2, dtype=dtype)
        self.dtype=dtype
        self.uses_learning_phase = True

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        if not hasattr(self, 'layer'):
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_output_at(i)
            regularized_loss += self.l1 * tensor.sum(tensor.mean(tensor.abs_(output), axis=0, dtype=self.dtype))
            regularized_loss += self.l2 * tensor.sum(tensor.mean(K.square(output), axis=0, dtype=self.dtype))
        return K.in_train_phase(regularized_loss, loss)      # Todo: replace all in_train_phase  [DV]

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}


def l1(l=0.01):
    return WeightRegularizer(l1=l)


def l2(l=0.01):
    return WeightRegularizer(l2=l)


def l1l2(l1=0.01, l2=0.01):
    return WeightRegularizer(l1=l1, l2=l2)


def activity_l1(l=0.01):
    return ActivityRegularizer(l1=l)


def activity_l2(l=0.01):
    return ActivityRegularizer(l2=l)


def activity_l1l2(l1=0.01, l2=0.01):
    return ActivityRegularizer(l1=l1, l2=l2)


def get(identifier, **kwargs):
    return globals().get(identifier)
