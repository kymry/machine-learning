import math
import numpy as np


def sigmoid_dx(z):
    return sigmoid(z) * sigmoid(1 - sigmoid_(z))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh_dx(z):
    return 1 - np.square(tanh(z), 2)

def tanh(z):
    x = np.exp(2*z)
    return (x - 1) / (x + 1)
