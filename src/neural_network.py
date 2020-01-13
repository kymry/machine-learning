import os
import numpy as np
from initialize_parameters import initialize_weights_or_biases

class neural_network():

    def __init__(self, inputs_shapes, num_layers):
        self.num_layers = num_layers
        self.inputs = initialize_weights_or_biases(num_layers, inputs_shapes)


    def __repr__(self):
        pass
