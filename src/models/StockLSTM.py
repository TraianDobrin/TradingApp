import torch.nn as nn
from src.models.PositionalEncoding import PositionalEncoding


class StockLSTM(nn.Module):
    def __init__(self, d_model, hidden_size, num_layers = 3, dropout = 0.05, sequence_length = 60):
        super(StockLSTM, self).__init__()
        self.d_model = d_model
        self.PE = PositionalEncoding(self.d_model, sequence_length)
        self.t = nn.LSTM(input_size = self.d_model, hidden_size = hidden_size, num_layers = num_layers,
                         batch_first = True, dropout = dropout)
        self.final = nn.Linear(in_features=hidden_size, out_features=1)
    def forward(self, x, return_hidden=False):
        x = self.PE(x)
        out, (h_n, c_n) = self.t(x)  # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]  # (batch, hidden_size)

        if return_hidden:
            return last_hidden      # for downstream usage
        else:
            return self.final(last_hidden)
