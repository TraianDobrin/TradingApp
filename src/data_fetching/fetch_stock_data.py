import argparse
import yfinance as yf
import os


def fetch_and_save(symbol: str, start: str, interval: str, format: str = "parquet"):
    dir_path = f"data/raw/interval-{interval}"
    os.makedirs(dir_path, exist_ok=True)

    print(f"Downloading {symbol} data ({start} -> present, {interval} interval)...")
    df = yf.download(symbol, start=start, interval=interval)

    if df.empty:
        print(f"⚠️ No data returned for {symbol}. Skipping.")
        return

    file_path = f"data/raw/interval-{interval}/{symbol}.{format}"

    if format == "csv":
        df.to_csv(file_path)
    elif format == "parquet":
        df.to_parquet(file_path)
    else:
        raise ValueError("Format must be csv or parquet")

    print(f"✅ Saved {symbol} to {file_path}")


def load_symbols_from_file(path: str):
    symbols = []
    with open(path, "r") as f:
        for line in f:
            # Support both "AAPL,MSFT,GOOG" and one-ticker-per-line
            parts = [s.strip() for s in line.replace("\n", "").split(",") if s.strip()]
            symbols.extend(parts)
    return symbols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock data using Yahoo Finance and save locally")
    parser.add_argument("--symbol", type=str, help="Single stock ticker (e.g., AAPL, AMZN)")
    parser.add_argument("--symbols_file", type=str, help="Path to file with list of tickers (comma-separated or one per line)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (e.g., 1d, 1h, 5m)")
    parser.add_argument("--start", type=str, required=True, help="Data start date (e.g. 2020-01-01)")
    parser.add_argument("--format", type=str, default="parquet", choices=["csv", "parquet"], help="File format to save")

    args = parser.parse_args()

    # Collect symbols
    symbols = []
    if args.symbol:
        symbols.append(args.symbol)
    if args.symbols_file:
        symbols.extend(load_symbols_from_file(args.symbols_file))

    if not symbols:
        raise ValueError("Please provide either --symbol or --symbols_file")

    # Download each stock
    for sym in symbols:
        fetch_and_save(sym, args.start, args.interval, args.format)
