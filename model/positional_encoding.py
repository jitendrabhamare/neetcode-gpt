import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        """
        Computes the Sinusoidal Positional Encoding matrix.
        """        
        # Initialize the positional encoding matrix
        PE = np.zeros((seq_len, d_model))

        # position shape: (seq_len, 1) - Column vector of positions
        position = np.arange(seq_len).reshape(-1, 1)
        
        # denom_term shape: (d_model/2,) - The scaling factor for each frequency
        # Creates the sequence: 2i = 0, 2, 4, ...
        denom_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
        
        # Broadcasting: (seq_len, 1) / (d_model/2,) -> (seq_len, d_model/2)
        # Apply sine to even indices (0, 2, 4...)
        PE[:, 0::2] = np.sin(position / denom_term)
        
        # Apply cosine to odd indices (1, 3, 5...)
        PE[:, 1::2] = np.cos(position / denom_term)
        
        return np.round(PE, 5)
