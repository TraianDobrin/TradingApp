import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import yaml

RAW_DIR = f"../../data/raw/interval-1d"
PROC_DIR = f"../../data/processed/interval-1d"
CONFIG_FILE_STOCK_FEATURES = f"../../configs/stock_features.yaml"
os.makedirs(PROC_DIR, exist_ok=True)
with open(CONFIG_FILE_STOCK_FEATURES, "r") as f:
    config = yaml.safe_load(f)

rolling_window = config["preprocessing"]["rolling_window_size"]
scalers = {}

def compute_log_returns(df: pd.DataFrame, stock) -> pd.DataFrame:
    """
    Compute log returns for all price columns and percentage change for volume.
    """
    df = df.copy()

    # Daily return
    df['Return'] = (df['Close'] - df['Open']) / df['Open']
    df["Target"] = df["Return"].shift(-1)

    # Moving averages for daily return
    df['Moving averages'] = df['Return'].rolling(rolling_window).mean()

    # Momentum: difference between current price and price n periods ago
    df['Momentum'] = df['Return'] - df['Return'].shift(rolling_window)

    # Volatility: rolling standard deviation
    df['Volatility'] = df['Return'].rolling(rolling_window).std()

    ema_windows = (5, 12, 26)
    # ---- EMA features ----
    for w in ema_windows:
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
        df[f'EMA_dist_{w}'] = (df['Close'] - df[f'EMA_{w}']) / df[f'EMA_{w}']

    # Short vs long EMA crossover (classic trend strength indicator)
    if len(ema_windows) >= 2:
        short, long = ema_windows[0], ema_windows[-1]
        df[f'EMA_diff_{short}_{long}'] = df[f'EMA_{short}'] - df[f'EMA_{long}']
        df[f'EMA_trend_strength'] = df[f'EMA_diff_{short}_{long}'] / df[f'EMA_{long}']

    # ---- Trend indicators ----
    # 1. Direction: +1 if uptrend, -1 if downtrend
    df['Trend_strength'] = df[f'EMA_diff_{short}_{long}'] / df[f'EMA_{long}']



    # 2. Trend persistence: consecutive up days
    df['Up'] = (df['Return'] > 0).astype(int)
    df['Consec_up'] = df['Up'] * (df['Up'].groupby((df['Up'] != df['Up'].shift()).cumsum()).cumcount() + 1)
    df['Consec_down'] = ((df['Return'] < 0).astype(int)) * \
                        (((df['Return'] < 0).astype(int)).groupby(((df['Return'] < 0).astype(int) != (
                            (df['Return'] < 0).astype(int)).shift()).cumsum()).cumcount() + 1)

    df['Date'] = pd.to_datetime(df['Date'])

    df['MonthSin'] = np.sin(df['Date'].dt.month/12*2*np.pi)
    df['MonthCos'] = np.cos(df['Date'].dt.month/12*2*np.pi)
    df['Year'] = df['Date'].dt.year
    df['DaySin'] = np.sin(df['Date'].dt.day/31*2*np.pi)
    df['DayCos'] = np.cos(df['Date'].dt.day/31*2*np.pi)

    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]


    for col in price_cols:
        if col in df.columns:
            df[col + ' change'] = df[col].pct_change()
            df[col + ' original'] = df[col]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['Adj Close'] = scaler.fit_transform(df[['Adj Close']])
    if stock == 'AMZN':
        print("da coaie exist")
    scalers[stock] = scaler
    joblib.dump(scaler, "../../data/scalers/" + stock + ".pkl")

    # Volume often has zeros, so use percentage change instead of log
    if "Volume" in df.columns:
        df["Volume"] = np.log(df["Volume"])

    df = df.dropna()
    return df

def process_all():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)

    for file in os.listdir(RAW_DIR):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace(".csv", "")
        print(f"Processing {symbol}...")

        # Load raw data
        df = pd.read_csv(os.path.join(RAW_DIR, file))

        # Compute log returns
        df_ret = compute_log_returns(df, symbol)

        # Save
        out_path = os.path.join(PROC_DIR, f"{symbol}.csv")
        df_ret.to_csv(out_path, index = False)
        print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    process_all()
