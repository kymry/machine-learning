import sys
import os
sys.path.append(os.getcwd() + '/src')
import initialize_parameters as ip
import loss_functions as lf
import activation_functions as af


class NeuralNetwork():

    def __init__(self, features, labels, nodes_per_layer=[3,4,1]):
        """ nodes_per_layer: array of nodes at each layer.
        """
        self.num_layers = len(nodes_per_layer)
        self.weights_biases = ip.initialize_weights_and_biases(nodes_per_layer)
        self.features = features
        self.labels = labels
        self.activation_hidden = af.sigmoid
        self.activation_hidden_dx = af.sigmoid_dx
        self.activation_output = af.softmax
        self.activation_output_dx = af.softmax_dx
        self.loss_function = lf.cross_entropy
        self.loss_function_dx = lf.cross_entropy_dx
        self.learning_rate = 0.005

    def set_hidden_layer_activation(self, name):
        try:
            self.activation_hidden, self.activation_hidden_dx =
                af.get_activation_function(name)
        except KeyError:
            print("Invalid activation function")


    def set_output_layer_activation(self, name):
        try:
            self.activation_output, self.activation_output_dx =
                af.get_activation_function(name)
        except KeyError:
            print("Invalid activation function")


    def set_loss_function(activation(self, name):
            try:
                self.loss_function, self.loss_function_derivative =
                 lf.get_loss_function(name)
            except KeyError:
                print("Invalid loss function")


    def __repr__(self):
        return self.weights_biases
