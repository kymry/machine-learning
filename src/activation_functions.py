import math
import numpy as np


def sigmoid_dx(Z):
    return sigmoid(Z) * sigmoid(1 - sigmoid(Z))

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def tanh_dx(Z):
    return 1 - np.square(tanh(Z), 2)

def tanh(Z):
    x = np.exp(2*Z)
    return (x - 1) / (x + 1)
