import numpy as np


class BaseLayer:
    def forward(self, input: np.ndarray):
        raise NotImplementedError

    def backward(self, output_error: np.ndarray, learning_rate: float = 0.01):
        raise NotImplementedError


class Dense(BaseLayer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward(self, input: np.ndarray):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_error: np.ndarray, learning_rate: float = 0.01):
        weights_error = np.dot(self.input.T, output_error)
        input_error = np.dot(output_error, self.weights.T)

        # Update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class Network:
    def __init__(self):
        self.layers: list[BaseLayer] = []
        self.loss_function: BaseLayer = None

    def add(self, layer: BaseLayer) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)

    def set_loss(self, loss_function) -> None:
        """Set the loss function for the network."""
        self.loss_function = loss_function

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, output_error: np.ndarray, learning_rate: float = 0.01) -> None:
        """Backward pass through all layers in reverse order."""
        for layer in reversed(self.layers):
            output_error = layer.backward(output_error, learning_rate)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float = 0.01):
        """Training loop for the network."""
        for epoch in range(epochs):
            loss = 0
            for sample in range(len(X)):
                predictions = X[sample]
                predictions = self.forward(predictions)

                # Calculate loss
                loss += self.loss_function.forward(predictions, y[sample])

                # Backward pass
                loss_grad = self.loss_function.backward(predictions, y[sample])
                self.backward(loss_grad, learning_rate)

            if epoch % 100 == 0:
                loss /= len(X)
                print(f'epoch {epoch + 1} {epochs} loss={loss}')

    def predict(self, X: np.ndarray):
        result = []
        for i in range(len(X)):
            output = X[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result
