import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        num_layers = len(weights)
        h = x
        for i in range(num_layers):
            # The @ symbol is Python's built-in matrix multiplication operator
            # It's a modern, clean way and handles batches gracefully
            h = h @ weights[i] + biases[i]  # Linear transformation, 
            if i < num_layers - 1:
                h = np.maximum(0, h)  # apply ReLU at each layer, except the last one
        
        return np.round(h, 5)
