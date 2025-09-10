import argparse
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import os

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=window).mean()
    ma_down = down.rolling(window=window, min_periods=window).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_and_clean(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")

    # Handle MultiIndex or weird columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    adj_close_col = next((c for c in df.columns if 'Adj Close' in c), None)
    if adj_close_col is None:
        raise KeyError("Adjusted Close column not found in downloaded data!")

    keep_cols = ['Open', 'High', 'Low', 'Close', adj_close_col, 'Volume']
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().ffill().bfill()

    # Feature engineering
    df["Returns"] = df[adj_close_col].pct_change()
    df["LogRet"] = np.log(df[adj_close_col]) - np.log(df[adj_close_col].shift(1))
    df["MA_10"] = df[adj_close_col].rolling(window=10).mean()
    df["MA_50"] = df[adj_close_col].rolling(window=50).mean()
    df["STD_20"] = df[adj_close_col].rolling(window=20).std()
    df["RSI_14"] = compute_rsi(df[adj_close_col], window=14)

    df = df.dropna()
    df.rename(columns={adj_close_col: "Adj Close"}, inplace=True)
    return df

def summary_stats(df: pd.DataFrame):
    return df[["Adj Close", "Returns", "LogRet"]].describe()

def save_clean_csv(df: pd.DataFrame, ticker: str, out_dir: str = "data") -> str:
    os.makedirs(out_dir, exist_ok=True)
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
    df = df.dropna(subset=['Adj Close'])
    filename = os.path.join(out_dir, f"{ticker}_cleaned.csv")
    df.to_csv(filename)
    return filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    df = fetch_and_clean(args.ticker, args.start, args.end)
    print(f"Downloaded and cleaned {len(df)} rows for {args.ticker}")
    print(summary_stats(df))
    out = save_clean_csv(df, args.ticker)
    print(f"Saved cleaned data to {out}")

if __name__ == "__main__":
    main()
