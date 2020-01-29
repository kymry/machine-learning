import sys
import os
sys.path.append(os.getcwd() + '/src')
import backward_propagation as bp
import forward_propagation as fp


def train_network(nn, epochs):
    """ Trains the neural network for epochs """
    for iter in range(epochs):
        cache = fp.forward_propagation(nn)
        bp.backward_propagation(nn, cache)
