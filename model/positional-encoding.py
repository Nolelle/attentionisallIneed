import torch
import torch.nn as nn
import math

# d_model is dimension of the embeddings (vector of size 512)
# seq_len is the max length of a sentence
# dropout is the rate of dropout, a regularization technique that specifies the probabiltiy of setting to zero units in the nn during training.


class PositionalEnconding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # formula to find the positional encodings refern to paper.
        # Create a matrix of shape (seq_len, d_model)
        positional_encoding = torch.zeros(seq_len, d_model)

        # Create a vector of shaoe(seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # Sin for even, Cos for odd
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
