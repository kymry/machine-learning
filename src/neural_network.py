import os
import numpy as np
from initialize_parameters import initialize_weights_or_biases, add_input_to_params

class neural_network():

    def __init__(self, input, layer_shapes, labels):
        """ params: W^l and b^l for each layer l
            input: training examples
        """
        self.num_layers = len(layer_shapes)
        self.params = initialize_weights_and_biases(layer_shapes)
        self.input = input


    def __repr__(self):
        pass
