import unittest
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')
import backward_propagation as bp
import forward_propagation as fp
import initialize_parameters as ip
import loss_functions as lf
import activation_functions as af

class TestBackwardPropagation(unittest.TestCase):

    def setUp(self):
        self.layer_shapes = {0: 3, 1:4, 2:1}
        self.num_layers = len(self.layer_shapes)
        self.labels = np.array([[2,4]])
        self.input = np.array(([1,1],[2,2],[3,3]))
        self.num_train_examples = 2
        self.params = ip.initialize_weights_and_biases(self.layer_shapes)
        self.cache = fp.forward_propagation(self.params, self.labels, self.input)
        self.params.update(self.cache)
        self.da_loss = lf.mean_squared_error_dx(self.labels, self.params['A'+str(2)])


    def test_calculate_dz_output_layer(self):
        dZ = bp.calculate_dZ(self.da_loss, self.cache['Z'+str(self.num_layers - 1)], af.sigmoid_dx)
        self.assertEqual(self.cache['Z'+str(self.num_layers-1)].shape, dZ.shape)

    def test_calculate_db_output_layer(self):
        


    def test_backward_propagation(self):
        pass


if __name__ == "__main__":
    unittest.main()
