import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):
    def __init__(self, esp: float = 10**-6) -> None:
        super().__init__()
        self.esp = esp
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeroes(1))  # Addition

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return std.alpha * (x - mean) / (std + self.eps) + self.bias
