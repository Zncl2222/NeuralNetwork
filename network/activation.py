import numpy as np

from .nn import BaseLayer


class Tanh(BaseLayer):
    def forward(self, input: np.ndarray):
        self.input = input
        return np.tanh(input)

    def backward(self, output_error: np.ndarray, learning_rate: float = 0.01):
        return (1 - np.tanh(self.input) ** 2) * output_error


class ReLU(BaseLayer):
    def forward(self, input: np.ndarray):
        self.input = input
        return np.maximum(0, self.input)

    def backward(self, output_error: np.ndarray, learning_rate: float = 0.01):
        return output_error * (self.input > 0)


class Sigmoid(BaseLayer):
    def forward(self, input: np.ndarray):
        self.input = input
        return 1 / (1 + np.exp(-input))

    def backward(self, output_error: np.ndarray, learning_rate: float = 0.01):
        sigmoid_output = self.forward(self.input)
        return output_error * sigmoid_output * (1 - sigmoid_output)
