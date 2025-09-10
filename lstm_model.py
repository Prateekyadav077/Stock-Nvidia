"""
lstm_model.py

Loads cleaned CSV (produced by data_prep_analysis.py), prepares sequences, trains an LSTM,
saves the trained model + scaler, and provides test set predictions for visualization.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_cleaned(ticker: str, start_date: str = None, end_date: str = None, data_dir: str = "data") -> pd.DataFrame:
    path = os.path.join(data_dir, f"{ticker}_cleaned.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned data not found at {path}. Run data_prep_analysis.py first.")
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
    df = df.dropna(subset=['Adj Close'])
    
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    return df

def create_sequences(values: np.ndarray, time_steps: int = 60):
    X, y = [], []
    for i in range(len(values) - time_steps):
        X.append(values[i:i+time_steps])
        y.append(values[i+time_steps])
    return np.array(X), np.array(y)

def build_model(time_steps: int = 60, units: int = 50, dropout: float = 0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def train_and_save(ticker: str, time_steps: int = 60, epochs: int = 30, batch_size: int = 32):
    # Load only historical training data
    df = load_cleaned(ticker, start_date="2018-01-01", end_date="2025-01-01")
    
    prices = df[['Adj Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices)

    X, y = create_sequences(scaled, time_steps=time_steps)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model(time_steps=time_steps)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path} and scaler to {scaler_path}")
    return model_path, scaler_path


def load_saved(ticker: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Train first using train_and_save().")
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_future(model, last_sequence: np.ndarray, n_steps: int, scaler) -> np.ndarray:
    seq = last_sequence.copy()
    preds = []
    for _ in range(n_steps):
        p = model.predict(seq.reshape(1, *seq.shape), verbose=0)
        preds.append(p[0,0])
        seq = np.roll(seq, -1)
        seq[-1] = p
    preds = np.array(preds).reshape(-1,1)
    return scaler.inverse_transform(preds)

def get_test_predictions(ticker: str, time_steps: int = 60):
    model, scaler = load_saved(ticker)
    df = load_cleaned(ticker)
    prices = df[['Adj Close']].values
    scaled = scaler.transform(prices)
    X, y = create_sequences(scaled, time_steps=time_steps)
    
    preds_scaled = model.predict(X)
    preds = scaler.inverse_transform(preds_scaled)
    actual = df['Adj Close'].values[time_steps:].reshape(-1,1)
    dates = df.index[time_steps:]
    return dates, actual.flatten(), preds.flatten()

if __name__ == "__main__":
  if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--time_steps", type=int, default=60)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--future_days", type=int, default=30, help="Days to predict after 2025-01-01")
    args = p.parse_args()

    # Train model on historical data
    train_and_save(args.ticker, time_steps=args.time_steps, epochs=args.epochs)

    # Predict future after 2025-01-01
    model, scaler = load_saved(args.ticker)
    df_future = load_cleaned(args.ticker, start_date="2018-01-01", end_date="2025-01-01")
    last_seq = scaler.transform(df_future[['Adj Close']].values)[-args.time_steps:]
    future_preds = predict_future(model, last_seq, n_steps=args.future_days, scaler=scaler)

    print("Future predictions after 2025-01-01:", future_preds.flatten())

