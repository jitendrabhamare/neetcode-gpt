import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        max_length = 0
        wordset = set()
        for sentence in positive + negative:
            words = sentence.split()
            max_length = max(max_length, len(words))
            for word in words:
                wordset.add(word)

        wordlist = sorted(list(wordset))
        wordmap = {}
        for i in range(len(wordlist)):
            wordmap[wordlist[i]] = float(i + 1)

        tensors = [[0.0 for _ in range(max_length)] for _ in range(len(positive) + len(negative))]

        i = 0
        for sentence in positive + negative:
            words = sentence.split()
            for j in range(len(words)):
                tensors[i][j] = wordmap[words[j]]
            i += 1

        encoded = torch.tensor(tensors)
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)













