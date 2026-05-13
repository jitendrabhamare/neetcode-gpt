import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        
        # 1. Forward 
        z = np.dot(x, w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))

        # 2. Chain rule
        error = y_hat - y_true
        sigmoid_deriv = y_hat * (1 - y_hat)
        delta = error * sigmoid_deriv

        # 3. Gradient:
        dL_dw = np.round(delta * x, 5)
        dL_db = np.round(delta, 5)
        return (dL_dw, dL_db)

