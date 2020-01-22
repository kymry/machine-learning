import numpy as np

def create_function_dictionary():
    dict = {'MeanSquaredError': mean_squared_error,
            'CrossEntropyLoss': cross_entropy_loss}
    return dict

def mean_squared_error(labels, predictions):
    """ labels : numpy ndarray
        predictions : numpy ndarray
    """
    differences = np.subtract(labels, predictions)
    squares = np.square(differences)
    return np.sum(squares) / (2*len(labels[0]))


def mean_squared_error_dx(labels, predictions):
    return np.subtract(labels, predictions)


def cross_entropy_loss(labels, predictions):
    """ labels : numpy ndarray
        predictions : numpy ndarray
    """
    num_training_examples = len(labels)
    sum_of_lossese = np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))
    loss = np.squeeze((-1/num_training_examples) * sum_of_losses)
    return loss

def cross_entropy_loss_dx(labels, prediction):
    pass
