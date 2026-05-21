import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        
        # Initialize W and b
        num_samples, num_features = X.shape
        W = np.zeros(num_features,)
        b = 0.0

        for _ in range(epochs):
            # Calculate y_hat and loss function L
            y_hat = X @ W + b
            L = np.mean((y_hat - y) ** 2)

            # Calculate gradients
            dW = (2.0 / num_samples) * (X.T @ (y_hat - y))
            db = 2.0 * np.mean(y_hat - y)

            # Update W and b
            W = W - lr * dW
            b = b - lr * db

        return (np.round(W, 5), np.round(float(b), 5))

