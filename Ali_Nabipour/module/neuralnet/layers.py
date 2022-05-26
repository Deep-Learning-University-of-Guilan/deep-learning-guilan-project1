import numpy as np


class FCLayer:
    def __init__(self, input_size, output_size, landa=0.00, random=True):
        self.landa = landa
        if random:
            # Xavier Glorot Initialization
            limit = np.sqrt(2 / float(input_size + output_size))
            self.weights = np.random.normal(
                0.0, limit, size=(input_size, output_size))
            self.bias = np.random.normal(0.0, limit, size=(1, output_size))
        else:
            self.weights = np.zeros((input_size, output_size)) + 1e-15
            self.bias = np.zeros((1, output_size)) + 1e-15

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error) + \
            (self.landa * (self.weights))

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class SigmoidLayer:
    def dsigmoid(self, input):
        return np.exp(-input) / (1 + np.exp(-input))**2

    def forward(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input))

    def backward(self, output_error, dummy):
        return output_error * self.dsigmoid(self.input)
