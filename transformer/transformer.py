import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, dim_model: int = 512, num_heads: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, bias: bool = True):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model=model_dim, num_heads=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        transformer_output = self.transformer_encoder(src)
        output = self.fc_out(transformer_output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, num_layers: int, dim_feedforward: int,
                 dropout: float, layer_norm_eps: float, bias: bool):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model=dim_model, num_heads=num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        layer_norm_eps=layer_norm_eps, bias=bias)
        self.transform


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dim_feedforward: int,
                 dropout: float, layer_norm_eps: float, bias: bool):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model=dim_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias)
        self.linear1 = nn.Linear(dim_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MultiheadAttention(nn.Module):
    def __init__(self, dim_embed: int, num_heads: int, dropout: float, bias: bool):
        super(MultiheadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads, dropout=dropout, bias=bias)

    def forward(self, query, key, value):
        return self.multihead_attn(query, key, value)[0]