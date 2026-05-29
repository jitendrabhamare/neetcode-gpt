import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:        
        # Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        combined = positive + negative
        dictionary = sorted({word for sentence in combined for word in sentence.split()})

        # Encode each sentence by replacing words with their IDs
        word_map = {dictionary[i]: float(i + 1) for i, word in enumerate(dictionary)}
        
        # Combine positive + negative into one list of tensors
        tensors = [torch.tensor([word_map[word] for word in sentence.split()]) for sentence in combined]

        # Pad shorter sequences with 0s
        return nn.utils.rnn.pad_sequence(tensors, batch_first=True)
