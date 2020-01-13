import numpy as np

def mean_squared_error(labels, predictions):
    """ labels : numpy ndarray
        predictions : numpy ndarray
    """
    differences = np.subtract(labels, predictions)
    squares = np.square(differences)
    return sum(squares) / len(labels)
