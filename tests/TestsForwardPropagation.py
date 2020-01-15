import unittest
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')
from forward_propagation import forward_propagation
from initialize_parameters import initialize_weights_and_biases

class TestForwardPropagation(unittest.TestCase):

    def setUp(self):
        self.layer_shapes = {0: 3, 1:4, 2:1}
        self.num_layers = len(self.layer_shapes)
        self.labels = np.array([[2,4]])
        self.input = np.array(([1,1],[2,2],[3,3]))
        self.num_train_examples = 2
        self.params = initialize_weights_and_biases(self.layer_shapes)

    def test_forward_propagation(self):
        cache = forward_propagation(self.params, self.labels, self.input)
        #print(cache)
        for l in range(1, self.num_layers):
            self.assertEqual(cache['Z' + str(l)].shape, (self.layer_shapes[l], self.num_train_examples))
            self.assertEqual(cache['A' + str(l)].shape, (self.layer_shapes[l], self.num_train_examples))


if __name__ == "__main__":
    unittest.main()
