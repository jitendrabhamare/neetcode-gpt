import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        
        super().__init__()
        
        # Lock seed for reproducible weight initialization
        torch.manual_seed(0)
        
        # 1. Feature Extractor / Hidden Layer
        # Projects the 784 raw pixels into a 512-dimensional feature space
        self.linear1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        
        # 2. Regularization
        # Randomly zeroes 20% of elements to prevent overfitting
        self.dropout = nn.Dropout(p=0.2)

        # 3. Classification Head
        # Maps the 512 features to the 10 digit classes (0-9)
        self.linear2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        # images shape: (batch_size, 784)

        torch.manual_seed(0)

        x = self.linear1(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)

        # Return the model's prediction to 4 decimal places
        return torch.round(x, decimals=4)
