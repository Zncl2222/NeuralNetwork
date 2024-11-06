import numpy as np


class MeanSquareError:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size
