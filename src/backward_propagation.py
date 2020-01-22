import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')


def back_propagation(nn, cache):
    """ nn: neural network to be trained
        cache = {W^l: ndarray,
                 b^l: ndarray,
                 Z^l: W^l * W^l-1 + b^l,
                 A^l: activation_function(Z^l)}
    """
    num_layers = nn.num_layers

    # compute gradient of loss function
    dA_loss = nn.loss_function(nn.labels, cache['A' + str(num_layers - 1)])
    dA_dict = {'dA' + str(num_layers-1): dA_loss}

    # compute gradients for each layer and update weight and bias
    for layer in reversed(range(1, num_layers)):
        dA_prev, dZ = activation_function_gradient(dA_dict['dA'+str(layer)], cache, layer, nn.activation_function)
        dA_dict['dA' + str(layer-1)] = dA_prev
        dW, db = linear_function_gradient(dZ, cache, layer)
        update_weights_biases(cache, dW, db, layer, nn.learning_rate)


def activation_function_gradient(dA, cache, layer, activation):
    """ cache: Z, A_prev, W, b for the current layer only
    """
    dZ = calculate_dZ(dA, cache['Z'+str(layer)], activation)
    dA_prev = calculate_dA_prev(cache['W'+str(layer)], dZ)
    return (dA_prev, dZ)


def calculate_dZ(dA, Z, activation):
    """ dA: gradient of activaiton function
        Z: current layer linear output
    """
    return np.multiply(dA, activation(Z))


def calculate_dA_prev(W_next, dZ):
    """ W_next: weight vector of next layer
        dZ: gradient of current layer linear output
    """
    return np.dot(W_next.T, dZ)


def linear_function_gradient(dZ, cache, layer):
    """ cache: Z, A_prev, W, b for current layer only
    """
    dW = calculate_dW(dZ, cache['A'+str(layer-1)])
    db = calculate_db(dZ)

    return (dW, db)


def calculate_dW(dZ, A_prev):
    """ dZ: gradient of current layer linear output
        A_prev: non-linear output of previous layer
    """
    num_training_examples = len(dZ[0])
    return np.dot(dZ, A_prev.T) / num_training_examples


def calculate_db(dZ):
    """ dZ: gradient of current layer linear output
    """
    num_training_examples = len(dZ[0])
    return np.sum(dZ, axis=1, keepdims=True) / num_training_examples


def update_weights_biases(cache, dW, db, layer, learning_rate):
    """ cache: Z, A, W, b
        layer: update weights and biases for this layer only
    """
    cache['W'+str(layer)] = cache['W'+str(layer)] - learning_rate * dW
    cache['b'+str(layer)] = cache['b'+str(layer)] - learning_rate * db
