import numpy as np

def initialize_w_b(num_nodes, num_variables):
    """ initializes the weights or biases for one layer
     with random numbers in [0.0, 1.0) """
    np.random.seed(19920)
    return np.random.rand(num_nodes, num_variables)
