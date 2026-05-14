import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # Convert inputs into np arrays
        x_arr = np.array(x)
        W1_arr = np.array(W1)
        b1_arr = np.array(b1)
        W2_arr = np.array(W2)
        b2_arr = np.array(b2)
        y_arr = np.array(y_true)
        
        # Forward pass
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0, z1)
        z2 = np.dot(W2_arr, a1) + b2_arr
        loss = np.mean((z2 - y_true) ** 2)

        # Backward pass
        n = len(y_arr)
        dz2 = (2.0 / n) * (z2 - y_arr)

        # 1. Layer 2
        dW2 = np.outer(dz2, a1)
        db2 = dz2

        # 2. Backpropagate to Layer 1
        da1 = np.dot(W2_arr.T, dz2)

        # 3. The ReLU derivative binary mask: True (1) if z1 > 0, else False (0)
        dz1 = da1 * (z1 > 0)

        # 4. Layer 1
        dW1 = np.outer(dz1, x_arr)
        db1 = dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': (np.round(dW1, 4) + 0.0).tolist(),
            'db1': (np.round(db1, 4) + 0.0).tolist(),
            'dW2': (np.round(dW2, 4) + 0.0).tolist(),
            'db2': (np.round(db2, 4) + 0.0).tolist(),
        }