import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim_model: int, dim_feedforward: int, dropout: float, bias: bool):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_model, dim_feedforward, bias=bias)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model, bias=bias)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))
