import math
from typing import Optional

import torch
import torch.nn as nn

from .embedding import Embedding
from .encodelayer import EncodeLayer


class Encoder(nn.Module):
    def __init__(self, num_embed:int, dim_model: int, num_heads: int, num_layers: int, dim_feedforward: int, dropout: float, bias: bool):
        r"""Encoder
            num_embed: vocab_size, 词汇表的总次数，这里设置为858
            dim_model: 每个词映射成的向量维度，这里设置为128
            num_heads: 头的个数
            num_layers: encoder层数
            dim_feedforward: 隐藏层维度，一般是dim_model * 4
            dropout: dropout
            bias: 是否设置偏置
        """
        super(Encoder, self).__init__()
        self.embedding = Embedding(num_embed, dim_model, dropout=dropout, pad_idx=-1)
        self.layers = nn.ModuleList([
            EncodeLayer(dim_model, num_heads, dim_feedforward, dropout, bias) for _ in range(num_layers)
        ])

    def forward(self, x):
        mask = (x == -1).unsqueeze(1).unsqueeze(2).expand(x.size(0), 1, x.size(1), x.size(1))

        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x
