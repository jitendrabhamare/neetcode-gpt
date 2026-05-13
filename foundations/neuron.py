import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        
        z = np.dot(x, w) + b
        if activation == "sigmoid":
            return np.round(1 / (1 + np.exp(-z)), 5)
        elif activation == "relu":
            return np.round(max(0, z) / 1.0, 5)
        else:
            return np.round(float(z), 5)
