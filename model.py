import math

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=16, num_encoder_layers=1, num_classes=17):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=-1)  # 将输入数字映射到高维空间
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # 添加位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)

        # Transformer编码
        output = self.transformer_encoder(x)

        # 分类输出
        output = output.mean(dim=1)  # 对序列长度进行全局平均池化
        output = self.fc_out(output)

        return torch.sigmoid(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
