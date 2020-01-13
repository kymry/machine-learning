import unittest
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')
from initialize_parameters import initialize_weights_and_biases

class TestInitializeParameters(unittest.TestCase):

    def setUp(self):
        self.layer_shapes = {0: 3, 1:4, 2:1}
        self.num_layers = len(self.layer_shapes)
        self.labels = np.array([2,4,6,8])
        self.input = np.array(([1,1],[2,2],[3,3]))
        self.params = initialize_weights_and_biases(self.layer_shapes)

    def test_weight_bias_initialization(self):
        #print(self.params)
        for l in range(1, self.num_layers):
            self.assertEqual(self.params['W' + str(l)].shape, (self.layer_shapes[l], self.layer_shapes[l-1]))
            self.assertEqual(self.params['b' + str(l)].shape, (self.layer_shapes[l-1], 1))


if __name__ == '__main__':
    unittest.main()
