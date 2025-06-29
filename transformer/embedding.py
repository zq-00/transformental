import math
from typing import Optional

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embed: int, embed_dim: int, max_len: int = 128,
                 dropout: float = 0.1, pad_idx: Optional[int] = None, ) -> None:
        super().__init__()
        self.embeding = nn.Embedding(num_embed, embed_dim, pad_idx=pad_idx)

        self.position = Positional(embed_dim, max_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.position(self.embeding(x)))

class Positional(nn.Module):
    def __init__(self, embed_dim: int, max_len: int) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, 'Embedding dimension must be even.'

        pe = torch.zeros(max_len, embed_dim).float()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
