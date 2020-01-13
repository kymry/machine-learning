import numpy as np

def initialize_weights_or_biases(num_layers, nodes_at_layer):
    """ initializes the weights and biases for all layers
     with random numbers in [0.0, 1.0)
     W.shape = [# nodes at layer, # nodes at previous layer ]
     """
    np.random.seed(19920)
    weights_biases = {}

    for layer in range(1, num_layers + 1):
        weights_biases['W' + str(layer)] = np.random.rand(nodes_at_layer[layer], nodes_at_layer[layer-1])
        weights_biases['b' + str(layer)] = np.random.rand(nodes_at_layer[layer-1], 1)

    return weights_biases
