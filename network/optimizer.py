import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import BaseLayer


class Optimizer:
    def update(
        self,
        layer: 'BaseLayer',
        grad_w: np.ndarray,
        grad_b: np.ndarray,
        learning_rate: float,
    ):
        raise NotImplementedError


class SGD(Optimizer):
    def update(
        self,
        layer: 'BaseLayer',
        grad_w: np.ndarray,
        grad_b: np.ndarray,
        learning_rate: float = 0.01,
    ):
        layer.weights -= learning_rate * grad_w
        layer.bias -= learning_rate * grad_b
