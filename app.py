"""
Streamlit app for NVIDIA Stock Predictor + News Sentiment + AI Assistant
"""

import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from lstm_model import create_sequences, predict_future, load_saved, get_test_predictions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
card_bg = "#2A2A2A"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {secondary_color};
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }}
    h1, h2, h3 {{
        color: {primary_color};
        font-weight: bold;
    }}
    .card {{
        background-color: {card_bg};
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #8CD600;
        color: black;
    }}
    .stNumberInput>div>input {{
        background-color: #2B2B2B;
        color: white;
        border-radius: 8px;
        border: 1px solid #444;
    }}
    .logo {{
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 15px;
    }}
    .logo img {{
        height: 60px;
        margin-right: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- APP HEADER -----------------
st.markdown(
    """
    <div class="logo">
        <img src="https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg">
        <h1>NVIDIA Stock Predictor + News Tracker</h1>
    </div>
    """,
    unsafe_allow_html=True
)

ticker = "NVDA"
st.sidebar.success(f"📈 Tracking Ticker: **{ticker}**")

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("⚙️ Settings")
time_steps = st.sidebar.number_input("Time steps (sequence length)", min_value=10, max_value=200, value=60)
future_days = st.sidebar.number_input("Days to predict (business days)", min_value=1, max_value=150, value=100)

# ----------------- LAYOUT: LEFT (Stocks/News) + RIGHT (AI Assistant) -----------------
left, right = st.columns([3, 1], gap="large")

# ----------------- LEFT SIDE CONTENT -----------------
with left:
    # HISTORICAL PREDICTION
    st.markdown('<div class="card">', unsafe_allow_html=True)
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
            st.success(f"📉 **RMSE:** {rmse:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # FUTURE PREDICTION
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🔮 Future Predictions")
    if st.button("Predict Future Prices"):
        try:
            model, scaler = load_saved(ticker)
        except FileNotFoundError as e:
            st.error(f"{e}\nRun data_prep_analysis.py and train the model first.")
        else:
            df = pd.read_csv(os.path.join("data", f"{ticker}_cleaned.csv"), index_col=0, parse_dates=True)
            scaled = scaler.transform(df.values)
            last_seq = scaled[-time_steps:]
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
            plt.tight_layout()
            st.pyplot(fig)

            out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred.flatten()})
            out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.date
            st.subheader("📅 Future Predicted Prices")
            st.dataframe(out_df)
    st.markdown('</div>', unsafe_allow_html=True)

    # NEWS & SENTIMENT
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📰 Latest NVIDIA News & Sentiment")

    API_KEY = "daa8c9fb223c44b2b8e6d38bb56835c7"  # your NewsAPI key
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "NVIDIA OR NVDA",
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 5,
        "apiKey": API_KEY
    }

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
                vs = analyzer.polarity_scores(combined_text)
                if vs['compound'] >= 0.05:
                    sentiment_str = "Positive ✅"
                elif vs['compound'] <= -0.05:
                    sentiment_str = "Negative ❌"
                else:
                    sentiment_str = "Neutral ⚪"
                st.markdown(f"### {title} ({sentiment_str})")
                st.write(description)
                st.markdown(f"[Read more]({url_link})")
                st.markdown("---")
        else:
            st.info("No recent NVIDIA news found.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- RIGHT SIDE CONTENT: AI ASSISTANT -----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Assistant")

    if "ai_open" not in st.session_state:
        st.session_state.ai_open = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("💬 Open AI Chat" if not st.session_state.ai_open else "❌ Close AI Chat"):
        st.session_state.ai_open = not st.session_state.ai_open

    if st.session_state.ai_open:
        st.markdown(
            """
            <style>
            .chat-box {
                background-color: #2B2B2B;
                border-radius: 12px;
                padding: 12px;
                height: 400px;
                overflow-y: auto;
                border: 1px solid #444;
                margin-bottom: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Show chat history
        st.markdown('<div class="chat-box">', unsafe_allow_html=True)
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"🧑 **You:** {msg}")
            else:
                st.markdown(f"🤖 **AI:** {msg}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Input + API call
        user_q = st.text_input("Ask something about stocks, investing, or NVIDIA:")

        if st.button("Send"):
            if user_q.strip():
                st.session_state.chat_history.append(("user", user_q))

                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=os.getenv(
                        "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"
                    ))

                    prompt = (
                        f"You are a financial assistant AI. Answer the following question in a clear, concise way:\n\n"
                        f"Question: {user_q}"
                    )

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=250
                    )

                    answer = response.choices[0].message.content
                    st.session_state.chat_history.append(("ai", answer))

                except Exception as e:
                    st.session_state.chat_history.append(("ai", f"Error: {e}"))

                st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("⚠️ Predictions are experimental. Do not use as financial advice.")
