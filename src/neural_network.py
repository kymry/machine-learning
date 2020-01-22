import sys
import os
sys.path.append(os.getcwd() + '/src')
import initialize_parameters as ip
import loss_functions as lf
import activation_functions as af


class NeuralNetwork():

    ACTIVATION_FUNCTIONS = af.create_function_dictionary()
    LOSS_FUNCTIONS = lf.create_function_dictionary()

    def __init__(self, features, labels, nodes_per_layer=[3,4,1]):
        """ nodes_per_layer: array of nodes at each layer.
        """
        self.num_layers = len(nodes_per_layer)
        self.weights_biases = ip.initialize_weights_and_biases(nodes_per_layer)
        self.features = features
        self.labels = labels
        self.learning_rate = 0.005
        self.activation_function = af.sigmoid
        self.loss_function = lf.mean_squared_error


    def set_hidden_layer_activation(self, function):
        try:
            self.hidden_layer_activation = NeuralNetwork.ACTIVATION_FUNCTIONS[function]
        except KeyError:
            print("Invalid activation function")


    def set_loss_function(activation(self, function):
            try:
                self.loss_function = NeuralNetwork.LOSS_FUNCTIONS[function]
            except KeyError:
                print("Invalid loss function")


    def __repr__(self):
        return self.weights_biases
