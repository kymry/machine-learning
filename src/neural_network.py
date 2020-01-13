import os
import numpy as np
import initialize_parameters

class neural_network():

    def __init__(self, num_hidden_layers=3):
        self.num_hidden_layers = num_hidden_layers
        #TODO - decide best way to accept W and B shapes as input parameters
        self.W = initialize_parameters.initialize_weights_or_biases(2, 3)
        self.B = initialize_parameters.initialize_weights_or_biases(2, 3)

    def __repr__(self):
        pass
