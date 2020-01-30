import numpy as np
import random as rn


class RegressionModel:

    def __init__(self, X=None, y=None, predict=None):
        self.X = X # mxn matrix. m training examples, n features
        self.y = y # 1xn vector of labels
        self.theta = np.zeros((1, X.shape[1])) # 1xn vector of weights
        self.num_examples = X.shape[0]

    def train(self, num_iters, train_rate):
        """ Stochastic Gradient Descent """
        for _ in range(num_iters):

            # train with random input each iteration
            training_examples = list(range(self.num_examples))
            rn.shuffle(training_examples)
            for i in training_examples[:5]:
                update = (self.y[i] - np.dot(self.theta, self.X[i])) * self.X[i]
                self.theta = self.theta + train_rate * update

    def cost(self):
        error = np.reshape(np.dot(self.X, self.theta.T), (1, self.num_examples)) - self.y
        return np.sum(error**2)

    def predict(self, x):
        return np.dot(self.theta, x)[0]
