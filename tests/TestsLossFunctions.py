import unittest
import numpy as np
import sys
import os
sys.path.append(os.getcwd() + '/src')
import loss_functions as lf

class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        self.labels = np.array([2,4,6,8])
        self.predictions = np.array([1,3,5,8])

    def test_mean_squared_error(self):
        loss = lf.mean_squared_error(self.labels, self.predictions)
        self.assertTrue(np.isclose(0.75, loss))


if __name__ == "__main__":
    unittest.main()
