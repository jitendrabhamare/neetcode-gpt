import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        '''Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)'''
        
        # Convert BOTH lists to a NumPy arrays
        x_arr = np.array(x, dtype=float)
        gamma_arr = np.array(gamma, dtype=float)

        # Calculate Root Mean Square (RMS)
        rms = np.sqrt(np.mean(x_arr ** 2) + eps)

        # Normalize (in-place division is memory-saving!)
        x_arr /= rms

        # Scale by gamma
        output = gamma_arr * x_arr

        return np.round(output, 4).tolist()
