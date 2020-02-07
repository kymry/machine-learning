import math
import numpy as np


def get_activation_function(name):
    if name not in ACTIVATIONS or name not in ACTIVATION_DERIVATIVES:
        raise KeyError
    else:
        return (ACTIVATIONS[name], ACTIVATION_DERIVATIVES[name])

def sigmoid_dx(Z):
    return sigmoid(Z) * sigmoid(1 - sigmoid(Z))

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def tanh_dx(Z):
    return 1 - np.square(tanh(Z), 2)

def tanh(Z):
    x = np.exp(2*Z)
    return (x - 1) / (x + 1)

def softmax(Z):
    return np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)), axis=0, keepdimes=True)

def softmax_dx(Z):
    num_classes = Z.shape[0]
    num_training_examples = Z.shape[1]
    np.matmul(Z, np.ones())


ACTIVATIONS = {'sigmoid': sigmoid, 'tanh': tanh}
ACTIVATION_DERIVATIVES = {'sigmoid', sigmoid_dx,'tanh', tanh_dx}
