import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        '''Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)'''
        
        # Convert Python list to a numpy array
        x_arr = np.array(x, dtype=float)

        # Normalize x
        rms = np.sqrt(np.mean(x_arr ** 2) + eps)
        x_arr /= rms

        # Scale by gamma
        output = gamma * x_arr

        return np.round(output, 4).tolist()


