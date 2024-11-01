import numpy as np


class BaseLayer:
    def forward(self, input: np.ndarray):
        raise NotImplementedError

    def backward(self, output_error: np.ndarray, learning_rate: float = 0.01):
        raise NotImplementedError


class Dense(BaseLayer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input: np.ndarray):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_error: np.ndarray, learning_rate: float = 0.01):
        weights_error = np.dot(output_error, self.input.T)
        input_error = np.dot(self.weights.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
