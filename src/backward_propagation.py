import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')


LEARNING_RATE = 0.005


def back_propagation(params, labels, loss_function_dx, activation_function_dx):
    """ params = {W^l: ndarray,
                  b^l: ndarray,
                  Z^l: W^l * W^l-1 + b^l,
                  A^l: activation_function(Z^l)}
        labels = true values of input data
    """

    num_layers = len(params) // 4
    dA_loss = loss_function_dx(labels, params['A' + str(num_layers)])

    for l in reversed(range(num_layers - 1)):
        cache = {'W': params['W'+str(l)], 'b': params['b'+str(l)],
                 'Z': params['Z'+str(l)], 'A_prev': params['A'+str(l-1)]}

        dA_prev, dZ = calculate_activation_function_dx(dA_loss, cache, activation_function_dx)
        dW, db = calculate_linear_function_dx(dZ, cache)
        update_params(cache, dW, db, LEARNING_RATE)


def calculate_activation_function_dx(dA, cache, activation_function_dx):
    """ cache: Z, A_prev, W, b for the current layer only
    """
    dZ = calculate_dZ(dA, cache['Z'], activation_function_dx)
    dA_prev = calculate_dA(cache['W'], dZ)
    return (dA_prev, dZ)


def calculate_linear_function_dx(dZ, cache):
    """ cache: Z, A_prev, W, b for current layer only
    """
    dW = calculate_dW(dZ, cache['A_prev'])
    db = calculate_db(dZ)

    return (dW, db)


def calculate_dZ(dA, Z, activation_function_dx):
    """ dA: gradient of activaiton function
        Z: current layer linear output
    """
    return np.multiply(dA, activation_function_dx(Z))


def calculate_dA(W_next, dZ):
    """ W_next: weight vector of next layer
        dZ: gradient of current layer linear output
    """
    return np.dot(W_next.T, dZ)


def calculate_dW(dZ, A_prev):
    """ dZ: gradient of current layer linear output
        A_prev: non-linear output of previous layer
    """
    num_training_examples = len(dZ[0])
    return np.dot(dZ, A_prez.T) / num_training_examples


def calculate_db(dZ):
    """ dZ: gradient of current layer linear output
    """
    num_training_examples = len(dZ[0])
    return np.sum(dZ, axis=1, keepdims=True) / num_training_examples


def update_params(cache, dW, db, learning_rate):
    """ cache: Z, A_prev, W, b for the current layer only
    """
    cache['W'] = cache['W'] - learning_rate * dW
    cache['b'] = cache['b'] - learning_Rate * db
