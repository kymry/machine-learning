import numpy as np
import activation_functions as af
import loss_functions as lf

def forward_propagation(num_layers, params, labels):
    """ Performs forward propagation while keeping a cache of the intermediate
        values Z and A.

        Z^l = W^l * W^l-1 + b^l
        A^l = activation_function(Z^l)
    """
    cache = {}

    # hidden layers
    for layer in range(1, num_layers):
        Z = np.dot(params['W'+str(layer)], params['W'+str(layer - 1)])
        A = af.sigmoid(Z)
        cache['Z' + str(layer)] = Z
        cache['A' + str(layer)] = A

    # output layer
    loss = lf.mean_squared_error(lables, cache['A' + str(num_layers)])
    cache["Loss"] = loss

    return cache
