import math
from typing import Optional

import torch
import torch.nn as nn

from embedding import Embedding
from encodelayer import EncodeLayer


class Encoder(nn.Module):
    def __init__(self, num_embed:int, dim_model: int, num_heads: int, num_layers: int, dim_feedforward: int, dropout: float, bias: bool):
        super(Encoder, self).__init__()
        self.embedding = Embedding(num_embed, dim_model, dropout=dropout, pad_idx=-1)
        self.layers = nn.ModuleList([
            EncodeLayer(dim_model, num_heads, dim_feedforward, dropout, bias) for _ in range(num_layers)
        ])

    def forward(self, x):
        mask = (x == 0).unsqueeze(1).unsqueeze(2).expand(x.size(0), 1, x.size(1), x.size(1))

        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x
