import unittest
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')
from initialize_parameters import initialize_weights_or_biases

class TestInitializeParameters(unittest.TestCase):

    def setUp(self):
        self.layer_sizes = {0: 5, 1:4, 2:4, 3:2, 4:1}
        self.num_layers = 4

    def test_weights_biases_shape(self):
        params = initialize_weights_or_biases(self.num_layers, self.layer_sizes)
        for l in range(1, self.num_layers + 1):
            self.assertEqual(params['W' + str(l)].shape, (self.layer_sizes[l], self.layer_sizes[l-1]))
            self.assertEqual(params['b' + str(l)].shape, (self.layer_sizes[l-1], 1))


if __name__ == '__main__':
    unittest.main()
