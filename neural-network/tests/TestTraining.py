import unittest
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.datasets import load_iris
import sys
import os
sys.path.append(os.getcwd() + '/src')
import backward_propagation as bp
import forward_propagation as fp
import initialize_parameters as ip
import loss_functions as lf
import activation_functions as af

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.iris = load_iris()
        self.data = self.iris.data[:,[0,1]]
        self.labels = self.iris.target
        self.num_training = self.labels.shape[0]

    def test_training_two_layer(self):
        formatter = plt.FuncFormatter(lambda i, *args: self.iris.target_names[int(i)])
        plt.figure(figsize=(5, 4))
        plt.scatter(self.iris.data[:, 0], self.iris.data[:, 1], c=self.iris.target)
        plt.colorbar(ticks=[0, 1, 2], format=formatter)
        plt.xlabel(self.iris.feature_names[0])
        plt.ylabel(self.iris.feature_names[1])

        plt.tight_layout()
        #plt.show()

if __name__ == "__main__":
    unittest.main()
