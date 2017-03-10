from lasagne.layers import Layer, MergeLayer
import lasagne.init as init
from theano import tensor
from collections import OrderedDict

#[DV] all 'int64' cast changed to 'int32'

__all__ = [
    "ExpressionLayer_Merge",
    "CenterLayer"
    # "DotLayer"
]

class ExpressionLayer_Merge(MergeLayer):
    """
    This layer provides boilerplate for a custom layer that applies theano function
    for input layers

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.

    function : callable
        A function to be applied to the output of the previous layer.

    output_shape : None, callable, tuple, or 'auto'
        Specifies the output shape of this layer. If a tuple, this fixes the
        output shape for any input shape (the tuple can contain None if some
        dimensions may vary). If a callable, it should return the calculated
        output shape given the input shape. If None, the output shape is
        assumed to be the same as the input shape. If 'auto', an attempt will
        be made to automatically infer the correct output shape.

    Notes
    -----
    An :class:`ExpressionLayer` that does not change the shape of the data
    (i.e., is constructed with the default setting of ``output_shape=None``)
    is functionally equivalent to a :class:`NonlinearityLayer`.

    Examples
    --------
    >>> from lasagne.layers import InputLayer, ExpressionLayer
    >>> l_in = InputLayer((32, 100, 20))
    >>> l1 = ExpressionLayer(l_in, lambda X: X.mean(-1), output_shape='auto')
    >>> l1.output_shape
    (32, 100)
    """
    def __init__(self, incomings, function, output_shape=None, **kwargs):
        super(ExpressionLayer_Merge, self).__init__(incomings, **kwargs)

        if output_shape is None:
            self._output_shape = None
        elif hasattr(output_shape, '__call__'):   # [DV] this is a function here
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.function = function

    def get_output_shape_for(self, input_shapes):
            return self._output_shape

    def get_output_for(self, inputs, **kwargs):
        return self.function(inputs)

class CenterLayer(MergeLayer):
    """
    Compute the class centers during training
    Ref. to "Discriminative feature learning approach for deep face recognition (2016)"
    """
    def __init__(self, incomings, Ncenter, alpha=0.9, centers=init.GlorotUniform(), **kwargs):
        """
        :param incomings: incomings[0] = features, incomings[1] = target labels
        :param alpha: moving averaging coefficient
        :param center: initial value of center
        :param kwargs:
        """
        super(CenterLayer, self).__init__(incomings, **kwargs)
        feature_dim = incomings[0].output_shape[-1]
        self.centers = self.add_param(centers, [Ncenter, feature_dim], name="centers")
        self._output_shape = [Ncenter, feature_dim]
        self.alpha = alpha
        self.updates = OrderedDict()

    def get_output_shape_for(self, input_shapes):
        return self._output_shape

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        if deterministic:
            return self.centers
        else:
            features, labels = inputs
            centers_batch = self.centers[labels,:]
            diff = (self.alpha - 1.0) * (centers_batch - features)

            centers_updated = tensor.inc_subtensor(self.centers[labels,:], diff)
            self.add_update([[self.centers, centers_updated]])
            return self.centers





# class DotLayer(MergeLayer):
#     """
#     This layer performing dot multiplication for input layers
#
#     Parameters
#     ----------
#     incomings :  a tuple of layers  feeding into this layer
#
#     function : callable
#         A function to be applied to the output of the previous layer.
#
#     """
#     def __init__(self, incomings, function, output_shape=None, **kwargs):
#         super(ExpressionLayer_Merge, self).__init__(incomings, **kwargs)
#
#         if output_shape is None:
#             self._output_shape = None
#         elif hasattr(output_shape, '__call__'):   # [DV] this is a function here
#             self.get_output_shape_for = output_shape
#         else:
#             self._output_shape = tuple(output_shape)
#
#         self.function = function
#
#     def get_output_shape_for(self, input_shapes):
#             return self._output_shape
#
#     def get_output_for(self, inputs, **kwargs):
#         return self.function(inputs)
