import torch
import torch.nn as nn

class OutputHead(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()

        self.linear = nn.Linear(
            embedding_dim,
            vocab_size
        )
    
    def forward(self, x):
        return self.linear(x)

outputhead = OutputHead