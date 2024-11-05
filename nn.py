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


class Tanh(BaseLayer):
    def forward(self, input: np.ndarray):
        self.input = input
        return np.tanh(input)

    def backward(self, output_error, learning_rate=0.01):
        return (1 - np.tanh(self.input) ** 2) * output_error


class MeanSquareError:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class Network:
    def __init__(self):
        self.layers: list[BaseLayer] = []
        self.loss_function = None

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


if __name__ == '__main__':
    # Generate AND gate data with noise
    np.random.seed(0)
    base_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    base_outputs = np.array([[0], [0], [0], [1]])

    # Expand dataset by adding noise around each base input
    x_train = []
    y_train = []
    samples_per_input = 50

    for i, base_input in enumerate(base_inputs):
        for _ in range(samples_per_input):
            noisy_input = base_input + 0.1 * np.random.randn(2)
            x_train.append(noisy_input)
            y_train.append(base_outputs[i])

    x_train = np.array(x_train).reshape(-1, 1, 2)
    y_train = np.array(y_train).reshape(-1, 1, 1)

    # Initialize the network
    nn = Network()
    nn.add(Dense(2, 9))
    nn.add(Tanh())
    nn.add(Dense(9, 1))
    nn.add(Tanh())

    # Set loss function and train the network
    nn.set_loss(MeanSquareError())
    nn.train(x_train, y_train, epochs=1000, learning_rate=0.1)

    # Test predictions on the base inputs
    predictions = nn.predict(base_inputs.reshape(-1, 1, 2))
    print('Predictions on base AND gate inputs:')
    for i, input_data in enumerate(base_inputs):
        print(f'Input: {input_data}, Prediction: {predictions[i][0]}, True: {base_outputs[i][0]}')
