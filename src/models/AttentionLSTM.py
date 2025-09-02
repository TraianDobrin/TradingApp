import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.StockLSTM import StockLSTM

class AttentionLSTM(nn.Module):
    def __init__(self, target_dim, other_dim, hidden_dim, rnn_hidden=64, num_layers=3, dropout=0.05):
        super(AttentionLSTM, self).__init__()
        # LSTM branch for target stock
        self.rnn = StockLSTM(target_dim, rnn_hidden, num_layers, dropout)

        # Attention layers
        self.query = nn.Linear(rnn_hidden, hidden_dim)
        self.key   = nn.Linear(other_dim, hidden_dim)
        self.value = nn.Linear(other_dim, hidden_dim)

        # Final prediction head
        self.final = nn.Linear(rnn_hidden + hidden_dim, 1)

    def forward(self, target_x, other_x):
        """
        target_x: (batch, seq_len, target_dim)    -> history of target stock
        other_x:  (batch, n_stocks, other_dim)    -> features of other stocks at prediction time
        """
        # Encode target stock sequence, get the final state
        rnn_out = self.rnn(target_x, return_hidden = True)  # (batch, rnn_hidden)

        # Project into attention space
        Q = self.query(rnn_out).unsqueeze(1)  # (batch, 1, hidden_dim)
        K = self.key(other_x)                 # (batch, n_stocks, hidden_dim)
        V = self.value(other_x)               # (batch, n_stocks, hidden_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # (batch, 1, n_stocks)
        attn_weights = F.softmax(attn_scores, dim=-1)                             # (batch, 1, n_stocks)
        context = torch.matmul(attn_weights, V).squeeze(1)                        # (batch, hidden_dim)

        # Concatenate RNN output + attention context
        combined = torch.cat([rnn_out, context], dim=-1)  # (batch, rnn_hidden + hidden_dim)

        # Final prediction
        out = self.final(combined)  # (batch, 1)
        return out, attn_weights.squeeze(1)   # also return weights to inspect relevance
