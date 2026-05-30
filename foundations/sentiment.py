import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        """Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid"""
        super().__init__()

        # Lock seed for reproducible weight initialization
        torch.manual_seed(0)

        # 1. Embedding Layer
        # Maps token IDs to dense 16-dimensional vectors
        self.embedding = nn.Embedding(vocabulary_size, 16)

        # 2. Classification Head
        # Maps the fixed-size 16D vector to a single score (logit)
        self.classfier = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Look up the vector for each token
        x = self.embedding(x)

        # "Bag of Words" Global Average Pooling
        # Collapse the sequence dimension (dim=1) to get one vector per sentence
        x = torch.mean(x, dim=1)

        # Get the probability score
        x = self.classfier(x)
        probs = self.sigmoid(x)

        return torch.round(probs, decimals=4)
