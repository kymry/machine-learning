import unittest
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')
from initialize_parameters import initialize_weights_and_biases, add_inputs_to_params

class TestInitializeParameters(unittest.TestCase):

    def setUp(self):
        self.layer_sizes = {0: 5, 1:4, 2:4, 3:2, 4:1}
        self.num_layers = len(self.layer_sizes)
        self.labels = np.array([2,4,6,8])
        self.params = initialize_weights_and_biases(self.layer_sizes)

    def test_weight_biase_shapes(self):
        print(self.params)
        for l in range(1, self.num_layers):
            self.assertEqual(self.params['W' + str(l)].shape, (self.layer_sizes[l], self.layer_sizes[l-1]))
            self.assertEqual(self.params['b' + str(l)].shape, (self.layer_sizes[l-1], 1))

    def test_add_lables(self):
        add_inputs_to_params(self.params, self.labels)
        print(self.params)


if __name__ == '__main__':
    unittest.main()
