import numpy as np

from src.models.AttentionLSTM import AttentionLSTM
import os
import pandas as pd
import torch
import torch.nn as nn
from trainers.AttentionLSTMTrainer import AttentionLSTMTrainer

RAW_DIR = f"../../data/raw/interval-1d"
PROC_DIR = f"../../data/processed/interval-1d"
os.makedirs(PROC_DIR, exist_ok=True)
all_stocks = {
    fname.replace(".csv", ""): pd.read_csv(os.path.join(PROC_DIR, fname))
    for fname in os.listdir(PROC_DIR)
    if fname.endswith(".csv")
}

features = ["Adj Close", "Volume"]
info_features = ['EMA_5', 'EMA_26', 'EMA_diff_5_26', 'Trend_strength']
target_feature = "Adj Close"
target = 'AMZN'
n_features = len(features)
n_info_features = len(info_features)

hidden_dim_attention_lstm = 64

attentionLSTM = AttentionLSTM(target_dim = n_features,
                              hidden_dim = hidden_dim_attention_lstm,
                              other_dim = n_info_features)

optimizerALSTM = torch.optim.Adam(attentionLSTM.parameters(), lr=1e-2)
criterionALSTM = torch.nn.L1Loss()


stocks_without_target = []
for ticker, df in all_stocks.items():
    if ticker != target:
        stocks_without_target.append(df[info_features].values)
window_size = 60
df_target = all_stocks[target]

# Target features: shape (samples, window, n_features)
X_np = np.lib.stride_tricks.sliding_window_view(df_target[features].values,
                                    (window_size, n_features))[:-1, 0, :, :]
Y_np = df_target[target_feature].values[window_size:]

# Context features: stack all other stocks -> shape (time, n_stocks-1, n_info_features)
other_arrays = [df[info_features].values for t, df in all_stocks.items() if t != target]

Z_all = np.stack(other_arrays, axis=1)  # (time, n_stocks-1, n_info_features)
Z_np = Z_all[window_size:]              # align with Y

# Convert to tensors
X = torch.tensor(X_np, dtype=torch.float32)                # (samples, window, n_features)
Y = torch.tensor(Y_np, dtype=torch.float32).unsqueeze(-1)  # (samples, 1)
Z = torch.tensor(Z_np, dtype=torch.float32)                # (samples, n_stocks-1, n_info_features)

attentionLSTMTrainer = AttentionLSTMTrainer(model = attentionLSTM,
                                            optimizer = optimizerALSTM,
                                            criterion = criterionALSTM,
                                            symbol = target,
                                            X = X,
                                            Y = Y,
                                            Z = Z)
attentionLSTMTrainer.fit(100, True, overfit_check=1)
loss, preds, targets, = attentionLSTMTrainer.validate()


