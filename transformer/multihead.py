import math
from typing import Optional

import torch
import torch.nn as nn

from attention import Attention


class MultiHead(nn.Module):
    def __init__(self, dim_embed: int, num_heads: int, dropout: float, bias: bool):
        super(MultiHead, self).__init__()
        assert dim_embed % num_heads == 0, 'Embedding dimension must be divisible by the number of heads.'

        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_heads

        self.query = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.key = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.value = nn.Linear(dim_embed, dim_embed, bias=bias)

        self.attention = Attention()
        self.linear = nn.Linear(dim_embed, dim_embed, bias=bias)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        x = self.attention(q, k, v, self.num_head, self.dim_head, mask=mask, dropout=self.dropout1)
        return self.dropout2(self.linear(x))
