import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        # Convert all Python lists to numpy arrays
        x_arr = np.array(x, dtype=float)
        gamma_arr = np.array(gamma, dtype=float)
        beta_arr = np.array(beta, dtype=float)
        r_mean = np.array(running_mean, dtype=float)
        r_var = np.array(running_var, dtype=float)

        if training:
            # Calculate batch statistics across rows (axis=0)
            mean = np.mean(x_arr, axis=0)
            var = np.mean((x_arr - mean) ** 2, axis=0)

            # Normalize
            x_hat = (x - mean) / np.sqrt(var + eps)

            # Update running statistics
            r_mean = np.multiply(1 - momentum, r_mean) + momentum * mean
            r_var = np.multiply(1 - momentum, r_var) + momentum * var
        else:
            # Inference mode: Use running statistics
            x_hat = (x - r_mean) / np.sqrt(r_var + eps)

        # Apply affine transform
        y = gamma * x_hat + beta

        # Round to 4 decimals and convert back to Python lists
        return (
            np.round(y, 4).tolist(),
            np.round(r_mean, 4).tolist(),
            np.round(r_var, 4).tolist()
        )


