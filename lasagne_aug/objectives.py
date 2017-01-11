import theano.tensor as tensor
from .ctc_theano import CTC_precise, CTC_for_train

__all__ = [
    "categorical_accuracy_log",
    "ctc_cost_for_train",
    "ctc_cost_precise"
]

def categorical_crossentropy_log(predictions, targets):
    """Computes the categorical cross-entropy between predictions and targets.

    .. math:: L_i = - \\sum_j{t_{i,j} \\log(p_{i,j})}

    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either targets in [0, 1] matching the layout of `predictions`, or
        a vector of int giving the correct class index per data point.

    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical cross-entropy.

    Notes
    -----
    This is the loss function of choice for multi-class classification
    problems and softmax output units. For hard targets, i.e., targets
    that assign all of the probability to a single class per data point,
    providing a vector of int for the targets is usually slightly more
    efficient than providing a matrix with a single 1.0 per row.
    """
    return theano.tensor.nnet.categorical_crossentropy(predictions, targets)

def ctc_cost_precise(seq, sm, seq_mask=None, sm_mask=None):
    """
    seq (B, L), sm (B, T, C+1), seq_mask (B, L), sm_mask (B, T)
    Compute CTC cost, using only the forward pass
    :param queryseq: (L, B)
    :param scorematrix: (T, C+1, B)
    :param queryseq_mask: (L, B)
    :param scorematrix_mask: (T, B)
    :param blank_symbol: scalar
    :return: negative log likelihood averaged over a batch
    """
    queryseq = seq.T
    scorematrix = sm.dimshuffle(1, 2, 0)
    if seq_mask is None:
        queryseq_mask = None
    else:
        queryseq_mask = seq_mask.T
    if sm_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = sm_mask.T

    return CTC_precise.cost(queryseq, scorematrix, queryseq_mask, scorematrix_mask)

def ctc_cost_for_train(seq, sm, seq_mask=None, sm_mask=None):
    """
    seq (B, L), sm (B, T, C+1), seq_mask (B, L), sm_mask (B, T)
    Compute CTC cost, using only the forward pass
    :param queryseq: (L, B)
    :param scorematrix: (T, C+1, B)
    :param queryseq_mask: (L, B)
    :param scorematrix_mask: (T, B)
    :param blank_symbol: scalar
    :return: negative log likelihood averaged over a batch
    """
    queryseq = tensor.addbroadcast(seq.T)
    scorematrix = sm.dimshuffle(1, 2, 0)
    if seq_mask is None:
        queryseq_mask = None
    else:
        queryseq_mask = seq_mask.T
    if sm_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = sm_mask.T

    return CTC_for_train.cost(queryseq, scorematrix, queryseq_mask, scorematrix_mask)

def ctc_best_path_decode(Y, Y_mask=None):
    """
    Decode the network output scorematrix by best-path-decoding scheme
    :param Y: output of a network, with shape (batch, timesteps, Nclass+1)
    :param Y_mask: mask of Y, with shape (batch, timesteps)
    :return:
    """
    scorematrix = Y.dimshuffle(1, 2, 0)
    if Y_mask is None:
        scorematrix_mask = None
    else:
        scorematrix_mask = Y_mask.dimshuffle(1, 0)
    blank_symbol = Y.shape[2] - 1
    resultseq, resultseq_mask = CTC_precise.best_path_decode(scorematrix, scorematrix_mask, blank_symbol)
    return resultseq, resultseq_mask

def ctc_CER(resultseq, targetseq, resultseq_mask=None, targetseq_mask=None):
    return CTC_precise.calc_CER(resultseq, targetseq, resultseq_mask, targetseq_mask)
