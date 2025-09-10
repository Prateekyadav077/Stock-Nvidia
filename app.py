"""
Streamlit app for NVIDIA Stock Predictor + News Sentiment
Uses preprocessed NVIDIA data and trained LSTM model.
Run with: streamlit run app.py
"""

import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from textblob import TextBlob
from lstm_model import create_sequences, predict_future, load_saved, get_test_predictions

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="NVIDIA Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🟢"
)

# ----------------- STYLE -----------------
primary_color = "#76B900"  # NVIDIA green
secondary_color = "#1D1D1B"  # Dark background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {secondary_color};
        color: white;
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 10px;
    }}
    .stNumberInput>div>input {{
        background-color: #2B2B2B;
        color: white;
        border-radius: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- APP TITLE -----------------
st.title("🟢 NVIDIA Stock Predictor + News Tracker")
ticker = "NVDA"
st.sidebar.markdown(f"**Ticker:** {ticker}")

# ----------------- USER INPUTS -----------------
time_steps = st.sidebar.number_input("Time steps (sequence length)", min_value=10, max_value=200, value=60)
future_days = st.sidebar.number_input("Days to predict (business days)", min_value=1, max_value=90, value=30)

# ----------------- HISTORICAL PREDICTION -----------------
st.header("📊 Historical Predictions")
if st.button("Show Historical vs Predicted"):
    try:
        dates, actual, predicted = get_test_predictions(ticker, time_steps=time_steps)
    except FileNotFoundError as e:
        st.error(f"{e}\nRun data_prep_analysis.py and train the model first.")
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, actual, color="lime", label="Actual")
        ax.plot(dates, predicted, color="cyan", linestyle="--", label="Predicted")
        ax.set_title(f"{ticker} — Historical vs Predicted Prices", color="white")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Adjusted Close Price", color="white")
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.legend()
        st.pyplot(fig)

        mse = ((actual - predicted) ** 2).mean()
        rmse = mse ** 0.5
        st.markdown(f"**RMSE:** {rmse:.4f}")

# ----------------- FUTURE PREDICTIONS -----------------
st.header("🔮 Future Predictions")
if st.button("Predict Future Prices"):
    try:
        model, scaler = load_saved(ticker)
    except FileNotFoundError as e:
        st.error(f"{e}\nRun data_prep_analysis.py and train the model first.")
    else:
        df = pd.read_csv(os.path.join("data", f"{ticker}_cleaned.csv"), index_col=0, parse_dates=True)
        prices = df["Adj Close"].values.reshape(-1, 1)
        scaled = scaler.transform(prices)
        X, y = create_sequences(scaled, time_steps=time_steps)

        last_seq = X[-1]
        future_pred = predict_future(model, last_seq, n_steps=int(future_days), scaler=scaler)

        last_date = df.index[-1]
        future_dates = []
        current = last_date
        while len(future_dates) < int(future_days):
            current += pd.Timedelta(days=1)
            if current.weekday() < 5:
                future_dates.append(current)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Adj Close"], color="lime", label="Actual")
        ax.plot(pd.to_datetime(future_dates), future_pred.flatten(), color="cyan", linestyle='--', label="Future Prediction")
        ax.set_title(f"{ticker} — Actual vs Future Prediction", color="white")
        ax.set_xlabel("Date", color="white")
        ax.set_ylabel("Adjusted Close Price", color="white")
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.legend()
        st.pyplot(fig)

        out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred.flatten()})
        out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.date
        st.subheader("Future Predicted Prices")
        st.dataframe(out_df)

# ----------------- NEWS + SENTIMENT -----------------
# ----------------- NEWS + SENTIMENT -----------------
st.header("📰 Latest NVIDIA News & Sentiment")
API_KEY = "daa8c9fb223c44b2b8e6d38bb56835c7"  # Replace with your NewsAPI key
url = "https://newsapi.org/v2/everything"
params = {
    "q": "NVIDIA OR NVDA",
    "sortBy": "publishedAt",
    "language": "en",
    "pageSize": 5,
    "apiKey": API_KEY
}

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

try:
    response = requests.get(url, params=params)
    news_data = response.json()
    if "articles" in news_data and len(news_data["articles"]) > 0:
        for article in news_data["articles"]:
            title = article.get("title", "")
            description = article.get("description", "")
            url_link = article.get("url", "")
            combined_text = f"{title}. {description}"

            # Analyze sentiment using VADER
            vs = analyzer.polarity_scores(combined_text)
            if vs['compound'] >= 0.05:
                sentiment_str = "Positive ✅"
            elif vs['compound'] <= -0.05:
                sentiment_str = "Negative ❌"
            else:
                sentiment_str = "Neutral ⚪"

            st.markdown(f"**{title}** ({sentiment_str})")
            st.write(description)
            st.markdown(f"[Read more]({url_link})")
            st.markdown("---")
    else:
        st.info("No recent NVIDIA news found.")
except Exception as e:
    st.error(f"Error fetching news: {e}")

st.markdown("---")
st.markdown(
    "**Notes:** NVIDIA Stock Predictor using LSTM. News sentiment is analyzed using TextBlob. "
    "Future predictions are for the next business days."
)
