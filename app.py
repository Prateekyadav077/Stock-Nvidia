# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import matplotlib.pyplot as plt
from lstm_model import train_model, create_sequences, build_model

st.title("📈 NVIDIA Stock Predictor + News Sentiment")

# Sidebar
ticker = "NVDA"
st.sidebar.write("Stock:", ticker)

# Load cleaned data
data = pd.read_csv(f"data/{ticker}_cleaned.csv", index_col=0, parse_dates=True)
st.subheader("📊 Historical Stock Data")
st.line_chart(data["Adj Close"])

# LSTM Predictions
st.subheader("🤖 Stock Price Prediction (LSTM)")
model, scaler = train_model(ticker, epochs=5)  # quick demo training

# Prepare data
close = data["Adj Close"].values.reshape(-1,1)
scaled = scaler.transform(close)
time_steps = 60
X, y = create_sequences(scaled, time_steps)
X = X.reshape((X.shape[0], X.shape[1],1))

preds = model.predict(X)
preds = scaler.inverse_transform(preds)

# Plot
fig, ax = plt.subplots()
ax.plot(data.index[time_steps:], close[time_steps:], label="Actual")
ax.plot(data.index[time_steps:], preds, label="Predicted")
ax.legend()
st.pyplot(fig)

# News + Sentiment
st.subheader("📰 Latest News")
API_KEY = "daa8c9fb223c44b2b8e6d38bb56835c7"
query = "NVIDIA OR NVDA AND stock OR market"

params = {
    "q": query,
    "sortBy": "publishedAt",
    "language": "en",
    "pageSize": 5,
    "apiKey": API_KEY
}

try:
    response = requests.get("https://newsapi.org/v2/everything", params=params)
    news = response.json()
    if "articles" in news:
        for article in news["articles"]:
            title = article["title"]
            desc = article["description"] or ""
            polarity = TextBlob(title + " " + desc).sentiment.polarity
            sentiment = "🟢 Positive" if polarity>0 else "🔴 Negative" if polarity<0 else "🟡 Neutral"
            st.markdown(f"**{title}** ({sentiment})")
            st.write(desc)
            st.write(f"[Read more]({article['url']})")
            st.write("---")
except Exception as e:
    st.error(f"Error fetching news: {e}")
