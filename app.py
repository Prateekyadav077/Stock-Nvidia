import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import datetime
from keras.models import load_model
from textblob import TextBlob

st.title("📈 NVIDIA Stock Predictor + News Sentiment Tracker")

ticker = "NVDA"
st.sidebar.write("Predicting stock for:", ticker)

start = datetime.date(2015, 1, 1)
end = datetime.date.today()

stock_data = yf.download(ticker, start=start, end=end)

st.subheader("📊 NVIDIA Stock Data")
st.write(stock_data.tail())
st.line_chart(stock_data["Close"])

st.subheader("🤖 Stock Price Prediction (LSTM Model)")
try:
    model = load_model("models/NVDA_lstm.h5")
    st.success("✅ Model loaded successfully. Predictions go here...")
except Exception as e:
    st.warning(f"⚠️ Could not load model: {e}")

st.subheader("📰 Latest News That May Affect NVIDIA Stock")

API_KEY = "daa8c9fb223c44b2b8e6d38bb56835c7"
url = "https://newsapi.org/v2/everything"
params = {
    "q": "NVIDIA",
    "sortBy": "publishedAt",
    "language": "en",
    "pageSize": 5,
    "apiKey": API_KEY
}

try:
    response = requests.get(url, params=params)
    news_data = response.json()
    if "articles" in news_data and len(news_data["articles"]) > 0:
        for article in news_data["articles"]:
            title = article['title']
            description = article['description'] if article['description'] else ""
            analysis = TextBlob(title + " " + description)
            polarity = analysis.sentiment.polarity
            if polarity > 0.05:
                sentiment = "🟢 Positive"
            elif polarity < -0.05:
                sentiment = "🔴 Negative"
            else:
                sentiment = "⚪ Neutral"
            st.markdown(f"**{title}**")
            st.write(description)
            st.caption(f"Sentiment: {sentiment}")
            st.write(f"[Read more]({article['url']})")
            st.caption(f"Published at: {article['publishedAt']}")
            st.write("---")
    else:
        st.info("No recent news found about NVIDIA.")
except Exception as e:
    st.error(f"Error fetching news: {e}")
