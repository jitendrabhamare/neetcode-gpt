import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        max_z = np.max(z)
        total = np.sum(np.exp(z - max(z)))
        return np.round((np.exp(z - max(z)) / total), 4)
