import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        """Computes mean, standard deviation, and dead neuron fraction for each Linear layer."""
        stats = []
        
        # Disable gradient tracking for pure inference diagnostics
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean_val = round(x.mean().item(), 4)
                    std_val = round(x.std().item(), 4)

                    # A neuron is "dead" only if it is <= 0 across the entire batch (dim=0)
                    if x.dim() >= 2:
                        dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        # Fallback for unbatched 1D inputs
                        dead_frac = round((x <= 0).float().mean().item(), 4)
                    
                    stats.append({"mean": mean_val, "std": std_val, "dead_fraction": dead_frac})

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        """Computes mean, standard deviation, and L2 norm of the weight gradients for each Linear layer."""
        model.zero_grad()

        # Calculate loss and backpropagate to populate module.weight.grad
        output = model(x)
        loss = nn.MSELoss()(output, y) # Forward + backward pass with nn.MSELoss
        loss.backward()
        
        stats = []

        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean_val = round(grad.mean().item(), 4)
                std_val = round(grad.std().item(), 4)
                norm_val = round(torch.norm(grad).item(), 4)

                stats.append({"mean": mean_val, "std": std_val, "norm": norm_val})

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        """Evaluates network health based on cascading thresholds."""
        
        # Check for widespread dead neurons (often caused by learning rate being too high with ReLU)
        for act_stat in activation_stats:
            if act_stat["dead_fraction"] > 0.5:
                return "dead_neurons"
    
        # Check for gradient explosions (often caused by poor initialization or missing normalization)
        for grad_stat in gradient_stats:
            if grad_stat["norm"] > 1000:
                return "exploding_gradients"

        # Check if the final layer gradients are vanishing (preventing the model from learning)
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # Check activation distributions across all layers
        for act_stat in activation_stats:
            if act_stat["std"] < 0.1:
                return "vanishing_gradients"
            elif act_stat["std"] > 10.0:
                return "exploding_gradients"

        # If all checks pass, the network architecture and initialization are stable
        return "healthy"

