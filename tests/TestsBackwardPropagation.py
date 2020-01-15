import unittest
import pprint
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

    def test_all_layers(self):
        pprint.pprint(self.params)
        gradients = bp.back_propagation(self.params, self.labels, self.num_layers, lf.mean_squared_error_dx, af.sigmoid_dx)
        pprint.pprint(self.params)

    def test_dz_output_layer(self):
        dZ = bp.calculate_dZ(self.da_loss, self.cache['Z'+str(self.num_layers - 1)], af.sigmoid_dx)
        self.assertEqual(self.cache['Z'+str(self.num_layers-1)].shape, dZ.shape)


    def test_db_output_layer(self):
        dZ = bp.calculate_dZ(self.da_loss, self.cache['Z'+str(self.num_layers - 1)], af.sigmoid_dx)
        db = bp.calculate_db(dZ)
        self.assertEqual(self.params['b'+str(self.num_layers - 1)].shape, db.shape)


    def test_dA_prev(self):
        dZ = bp.calculate_dZ(self.da_loss, self.cache['Z'+str(self.num_layers - 1)], af.sigmoid_dx)
        dA_prev = bp.calculate_dA_prev(self.params['W'+str(self.num_layers - 1)], dZ)
        self.assertEqual(dA_prev.shape, self.params['A'+str(self.num_layers-2)].shape)


    def test_dW(self):
        dZ = bp.calculate_dZ(self.da_loss, self.cache['Z'+str(self.num_layers - 1)], af.sigmoid_dx)
        dW = bp.calculate_dW(dZ, self.params['A'+str(self.num_layers-2)])
        self.assertEqual(dW.shape, self.params['W'+str(self.num_layers-1)].shape)


    def test_db(self):
        dZ = bp.calculate_dZ(self.da_loss, self.cache['Z'+str(self.num_layers - 1)], af.sigmoid_dx)
        db = bp.calculate_db(dZ)
        self.assertEqual(db.shape, self.params['b'+str(self.num_layers-1)].shape)

    def test_backward_propagation(self):
        pass


if __name__ == "__main__":
    unittest.main()
