import numpy as np


def generate_sin_test_data(num_training_examples):
    """ Generate test data from sin(x) """

    np.random.seed(1)
    input = np.random.uniform(-5, 5, num_training_examples)
    labels = np.array(np.sin(input))

    return (input, labels)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print(generate_sin_test_data(4))
