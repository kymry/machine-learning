import numpy as np


def get_loss_function(name):
    if name not in LOSS_FUNCTIONS or name not in LOSS_DERIVATIVES:
        raise KeyError
    else:
        return (LOSS_FUNCTIONS[name], LOSS_DERIVATIVES[name])

def mean_squared_error(labels, predictions):
    """ labels : numpy ndarray
        predictions : numpy ndarray
    """
    differences = np.subtract(labels, predictions)
    squares = np.square(differences)
    return np.sum(squares) / (2*len(labels[0]))


def mean_squared_error_dx(labels, predictions):
    return np.subtract(labels, predictions)


def cross_entropy(labels, predictions):
    """ labels : numpy ndarray
        predictions : numpy ndarray
    """
    num_classes = labels.shape[1]
    sum_of_losses = np.sum(lables * np.log(predictions))
    loss = np.squeeze((-1/num_classes) * sum_of_losses)
    return loss


def cross_entropy_dx(labels, predictions):
    return np.divide(labels, predictions) - np.divide(1 - labels, 1 - predictions)


LOSS_FUNCTIONS = {'MeanSquaredError': mean_squared_error,
                  'CrossEntropy': cross_entropy}

LOSS_DERIVATIVES = {'MeanSquaredError': mean_squared_error_dx,
                    'CrossEntropy': cross_entropy_dx}
