import math
from typing import Optional

import torch
import torch.nn as nn

from feedforward import FeedForward
from multihead import MultiHead


class EncodeLayer(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dim_feedforward: int, dropout: float, bias: bool):
        super(EncodeLayer, self).__init__()
        self.multihead = MultiHead(dim_model, num_heads, dropout, bias)
        self.feedforward = FeedForward(dim_model, dim_feedforward, dropout, bias)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.multihead(x, mask=mask))

        return self.norm2(x + self.feedforward(x))
