import unittest
import numpy as np
import math
import sys
import os
sys.path.append(os.getcwd() + '/src')
import activation_functions as af
import initialize_parameters as ip

class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid(self):
        a = af.sigmoid(np.array([0,1,2,3]))
        b = np.array([(1 / (1 + math.exp(-x))) for x in range(4)])
        self.assertTrue((a==b).all())

    def test_sigmoid_dx(self):
        pass

    def test_tanh(self):
        a = af.tanh(np.array([0,1,2,3]))
        b = np.tanh([0,1,2,3])
        self.assertTrue((a==b).all())

    def test_tanh_dx(self):
        pass

class TestInitializeParameters(unittest.TestCase):

    def test_initialize_w_b(self):
        pass


if __name__ == '__main__':
    unittest.main()
