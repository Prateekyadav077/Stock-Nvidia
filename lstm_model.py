# lstm_model.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import argparse
import os

def create_sequences(values, time_steps=60):
    X, y = [], []
    for i in range(len(values) - time_steps):
        X.append(values[i:i+time_steps])
        y.append(values[i+time_steps])
    return np.array(X), np.array(y)

def build_model(time_steps=60, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(time_steps,1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def train_model(ticker, epochs=15, time_steps=60):
    path = f"data/{ticker}_cleaned.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    close = df["Adj Close"].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(close)

    X, y = create_sequences(scaled, time_steps)
    X = X.reshape((X.shape[0], X.shape[1],1))

    split = int(0.8*len(X))
    X_train, y_train = X[:split], y[:split]

    model = build_model(time_steps)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{ticker}_lstm.h5"
    model.save(model_path)
    print(f"Saved trained model to {model_path}")
    return model, scaler
