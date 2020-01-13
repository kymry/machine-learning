import unittest
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')
from forward_propagation import forward_propagation
from initialize_parameters import initialize_weights_and_biases, add_inputs_to_params

class TestForwardPropagation(unittest.TestCase):

    def setUp(self):
        self.layer_shapes = {0:5, 1:4, 2:4, 3:2, 4:1}
        self.labels = np.array([2,4,6,8])
        self.inputs = np.array([2,3,1,2,3])
        self.params = initialize_weights_and_biases(self.layer_shapes)
        add_inputs_to_params(self.params, self.inputs)

    def test_forward_propagation(self):
        cache = forward_propagation(self.params, self.labels)
        print(cache)

if __name__ == "__main__":
    unittest.main()
