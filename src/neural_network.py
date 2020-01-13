import os
import numpy as np
from initialize_parameters import initialize_weights_or_biases, add_labels

class neural_network():

    def __init__(self, params, labels):
        self.num_layers = len(params)
        self.params = initialize_weights_and_biases(params)
        add_labels_to_params(params, labels)


    def __repr__(self):
        pass
