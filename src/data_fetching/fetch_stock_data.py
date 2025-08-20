import argparse
import yfinance as yf
import os

def fetch_and_save(symbol: str, start: str, interval: str, format: str = "parquet"):

    os.makedirs("data/raw", exist_ok=True)

    print(f"Downloading {symbol} data ({start} -> present, {interval} interval)...")
    df = yf.download(symbol, start = start, interval = interval)


    if df.empty:
        print("No data returned. Check symbol or params.")
        return

    file_path = f"data/raw/{symbol}_{interval}.{format}"

    if format == "csv":
        df.to_csv(file_path)
    elif format == "parquet":
        df.to_parquet(file_path)
    else:
        raise ValueError("Format must be csv or parquet")

    print(f"✅ Saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock data using Yahoo Finance and save locally")
    parser.add_argument("--symbol", type=str, required = True, help="Stock ticker (e.g., AAPL, AMZN)")
    parser.add_argument("--period", type=str, default = "1d", help="Data interval length (e.g., 1d, 5d, 1mo, 1y, max)")
    parser.add_argument("--start", type=str, required = True, help="Data start date (e.g. 2025-08-20)")
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet"], help="File format to save")

    args = parser.parse_args()
    fetch_and_save(args.symbol, args.start, args.period, args.format)
