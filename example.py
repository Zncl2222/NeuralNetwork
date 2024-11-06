import numpy as np

from network.nn import Network, Dense
from network.loss import MeanSquareError
from network.activation import Tanh


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
