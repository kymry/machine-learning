import numpy as np
import activation_functions as af
import loss_functions as lf


def forward_propagation(params, labels):
    """ Performs forward propagation while keeping a cache of the intermediate
        values Z and A.

        params = {W^l: ndarray, b^l: ndarray} 
        Z^l = W^l * W^l-1 + b^l
        A^l = activation_function(Z^l)
    """
    cache = {}
    num_layers = len(params) // 2

    # hidden layers
    for layer in range(1, num_layers + 1):
        Z = np.dot(params['W' + str(layer)], params['W'+str(layer - 1)])
        A = af.sigmoid(Z)
        cache['Z' + str(layer)] = Z
        cache['A' + str(layer)] = A

    # output layer
    loss = lf.mean_squared_error(labels, cache['A' + str(num_layers -  1)])
    cache["Loss"] = loss

    return cache
