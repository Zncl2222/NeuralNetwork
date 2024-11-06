import numpy as np

from .nn import BaseLayer


class Tanh(BaseLayer):
    def forward(self, input: np.ndarray):
        self.input = input
        return np.tanh(input)

    def backward(self, output_error, learning_rate=0.01):
        return (1 - np.tanh(self.input) ** 2) * output_error
