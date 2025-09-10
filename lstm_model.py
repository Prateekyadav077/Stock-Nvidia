import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load cleaned data with features
def load_cleaned_features(ticker: str, start_date: str = None, end_date: str = None, data_dir: str = "data") -> pd.DataFrame:
    path = os.path.join(data_dir, f"{ticker}_cleaned.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned data not found at {path}. Run data_prep_analysis.py first.")
    
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna()
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df

# Create multi-feature sequences
def create_sequences(values: np.ndarray, time_steps: int = 60):
    X, y = [], []
    for i in range(len(values) - time_steps):
        X.append(values[i:i+time_steps])
        y.append(values[i+time_steps, 0])  # predict 'Adj Close' only
    return np.array(X), np.array(y)

# Build improved LSTM
def build_model(time_steps: int, features: int, units: int = 128, dropout: float = 0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(time_steps, features)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

# Train model
def train_and_save(ticker: str, time_steps: int = 60, epochs: int = 100, batch_size: int = 32):
    df = load_cleaned_features(ticker, start_date="2018-01-01", end_date="2025-01-01")
    features = df.shape[1]  # Number of features for LSTM input
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)

    X, y = create_sequences(scaled, time_steps)
    split1 = int(len(X)*0.7)
    split2 = int(len(X)*0.9)
    X_train, X_val, X_test = X[:split1], X[split1:split2], X[split2:]
    y_train, y_val, y_test = y[:split1], y[split1:split2], y[split2:]

    model = build_model(time_steps, features)
    ...


# Load saved model
def load_saved(ticker: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.joblib")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or scaler not found. Train first.")
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Multi-step future prediction
def predict_future(model, last_sequence: np.ndarray, n_steps: int, scaler) -> np.ndarray:
    seq = last_sequence.copy()
    preds = []
    for _ in range(n_steps):
        p = model.predict(seq.reshape(1, *seq.shape), verbose=0)
        preds.append(p[0,0])
        seq = np.roll(seq, -1)
        seq[-1, 0] = p  # only update 'Adj Close'
    preds = np.array(preds).reshape(-1,1)
    # Scale back
    dummy = np.zeros((preds.shape[0], scaler.scale_.shape[0]))
    dummy[:,0] = preds[:,0]
    inv = scaler.inverse_transform(dummy)[:,0].reshape(-1,1)
    return inv

# Get test set predictions
def get_test_predictions(ticker: str, time_steps: int = 60):
    model, scaler = load_saved(ticker)
    df = load_cleaned_features(ticker)
    scaled = scaler.transform(df.values)
    X, y = create_sequences(scaled, time_steps)
    preds_scaled = model.predict(X)
    dummy = np.zeros((preds_scaled.shape[0], scaled.shape[1]))
    dummy[:,0] = preds_scaled[:,0]
    preds = scaler.inverse_transform(dummy)[:,0]
    actual = df['Adj Close'].values[time_steps:]
    dates = df.index[time_steps:]
    return dates, actual, preds

# Command-line interface
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="NVDA")
    p.add_argument("--time_steps", type=int, default=60)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--future_days", type=int, default=30)
    args = p.parse_args()

    train_and_save(args.ticker, time_steps=args.time_steps, epochs=args.epochs)

    model, scaler = load_saved(args.ticker)
    df_future = load_cleaned_features(args.ticker, start_date="2018-01-01", end_date="2025-01-01")
    last_seq = scaler.transform(df_future.values)[-args.time_steps:]
    future_preds = predict_future(model, last_seq, n_steps=args.future_days, scaler=scaler)

    print("Future predictions after 2025-01-01:", future_preds.flatten())
