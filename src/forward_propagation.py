import numpy as np
import activation_functions as af
import loss_functions as lf


def forward_propagation(params, labels, input):
    """ Performs forward propagation while keeping a cache of the intermediate
        values Z and A.

        params = {W^l: ndarray, b^l: ndarray}
        cache = {Z^l: W^l * W^l-1 + b^l, A^l: activation_function(Z^l)}
        input = {W^0: ndarray}
    """
    cache = {'A0': input}
    num_layers = len(params) // 2

    process_hidden_layers(cache, params, num_layers)
    compute_loss(labels, cache, num_layers)

    return cache

def process_hidden_layers(cache, params, num_layers):
        for layer in range(1, num_layers + 1):
            Z = np.dot(params['W' + str(layer)], cache['A' + str(layer - 1)])
            A = af.sigmoid(Z)
            cache['Z' + str(layer)] = Z
            cache['A' + str(layer)] = A


def compute_loss(labels, cache, num_layers):
    cache["Loss"] = lf.mean_squared_error(labels, cache['A' + str(num_layers)])
