import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        """
        Forward passes input through the model and calculates the fraction of 
        dead neurons (permanently zero) for each ReLU layer.
        """
        dead_fractions = []

        # Disable gradient tracking as this is purely diagnostic
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.ReLU):
                    # A neuron is dead if it outputs 0 for ALL samples in the batch (dim=0)
                    if x.dim() >= 2:
                        dead = (x == 0).all(dim=0).float().mean().item()
                    else:
                        # Fallback: if unbatched, check if the neuron is 0 for this single sample
                        dead = (x == 0).float().mean().item()
                    
                    dead_fractions.append(round(dead, 4))

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        """
        Diagnoses the health of ReLU layers and suggests architectural or 
        hyperparameter fixes based on the distribution of dead neurons.
        """
        # Edge case: No ReLU layers to evaluate
        if len(dead_fractions) == 0:
            return "healthy"

        max_fraction = max(dead_fractions)

        # 1. Catastrophic failure: over half a layer is dead. 
        # Switching activation functions is the safest recovery.
        if max_fraction > 0.5:
            return "use_leaky_relu"

        # 2. Early failure: if the first layer dies immediately, it's 
        # usually a sign of bad weight initialization (e.g., weights too large).
        if dead_fractions[0] > 0.3:
            return "reinitialize"

        # 3. Cascading failure: the learning rate might be pushing biases deeply 
        # negative over time, causing deeper layers to sequentially die.
        increasing = True
        for i in range(len(dead_fractions) - 1):
            if dead_fractions[i + 1] <= dead_fractions[i]:
                increasing = False
                break

        # Must have at least 2 layers to establish a trend of cascading deeper
        if len(dead_fractions) > 1 and increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"

        # 4. Normal operation: minor dead neuron counts (< 10%) are standard in ReLU networks
        if max_fraction < 0.1:
            return "healthy"

        # 5. Fallback: if it doesn't meet any severe critical thresholds, assume healthy
        return "healthy"

