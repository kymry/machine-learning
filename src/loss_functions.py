import numpy as np

def mean_squared_error(actual, predicted):
    """ actual : numpy ndarray
        predicted : numpy ndarray """
    difference_vector = np.subtract(actual, predicted)
    squared_vector = np.square(difference_vector)
    return sum(squared_vector) / len(actual)
