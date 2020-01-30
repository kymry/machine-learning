import unittest
import numpy as np
from linear_regression import RegressionModel

EPSILON = 0.1

class testRegression(unittest.TestCase):

    def test_1(self):
        x_0 = np.array([1] * 10)
        x_1 = np.array([i for i in range(10)])
        X = np.array([[x_0[i], x_1[i]] for i in range(10)])
        y = np.array([i*2 for i in range(10)])

        model = RegressionModel(X, y)
        model.train(1000, 0.01)

        price = np.array([1, 25])
        prediction = model.predict(price)
        self.assertTrue(price[1]*(2-EPSILON) <= prediction <= price[1]*(2+EPSILON))


if __name__ == '__main__':
    unittest.main()
