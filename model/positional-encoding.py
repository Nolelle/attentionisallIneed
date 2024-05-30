import torch
import torch.nn as nn
import math


class PositionalEnconding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # Create a matrix of shape (seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model)
