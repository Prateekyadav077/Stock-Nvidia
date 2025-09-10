# data_prep_analysis.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_and_clean(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df.dropna()
    df["Returns"] = df["Adj Close"].pct_change()
    df["LogRet"] = np.log(df["Adj Close"]) - np.log(df["Adj Close"].shift(1))
    df["MA_10"] = df["Adj Close"].rolling(10).mean()
    df["MA_50"] = df["Adj Close"].rolling(50).mean()
    df["STD_20"] = df["Adj Close"].rolling(20).std()
    df["RSI_14"] = compute_rsi(df["Adj Close"], 14)
    df = df.dropna()
    return df

def save_csv(df, ticker):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", f"{ticker}_cleaned.csv")
    df.to_csv(path)
    print(f"Saved cleaned data to {path}")
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    df = fetch_and_clean(args.ticker, args.start, args.end)
    save_csv(df, args.ticker)
