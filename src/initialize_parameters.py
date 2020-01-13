import numpy as np

def initialize_weights_and_biases(nodes_at_layer):
    """ initializes the weights and biases for all layers
        with random numbers in [0.0, 1.0)

        nodes_at_layer = {layer: # nodes at layer}
        W.shape = (# nodes at layer, # nodes at previous layer)
        b.shape = (# nodes at layer, 1)
     """
    np.random.seed(19920)
    params = {}

    for layer in range(1, len(nodes_at_layer)):
        params['W' + str(layer)] = np.random.rand(nodes_at_layer[layer], nodes_at_layer[layer-1])
        params['b' + str(layer)] = np.random.rand(nodes_at_layer[layer-1], 1)

    return params
