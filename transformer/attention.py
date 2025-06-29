import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def forward(self, query, key, value, num_head, dim_head, mask=None, dropout=None):
        query = query.view(query.size(0), query.size(1), num_head, dim_head).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), num_head, dim_head).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), num_head, dim_head).transpose(1, 2)

        atten = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            atten = atten.masked_fill(mask, -1e9)

        atten = F.softmax(atten, dim=-1)

        if dropout is not None:
            atten = dropout(atten)

        return torch.matmul(atten, value).transpose(1, 2).contiguous().view(query.size(0), query.size(1), num_head * dim_head)
