"""
Streamlit app for NVIDIA Stock Predictor (Interactive with Plotly)
"""

import streamlit as st
import pandas as pd
import os
from lstm_model import create_sequences, predict_future, load_saved, get_test_predictions
import plotly.graph_objects as go

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="NVIDIA Stock Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
    page_icon="🟢"
)

# ----------------- THEME COLORS -----------------
primary_color = "#76B900"  # NVIDIA green
secondary_color = "#121212"  # Dark background
card_bg = "#1E1E1E"

# ----------------- GLOBAL STYLING -----------------
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
        font-weight: 600;
        text-align: center;
    }}
    .card {{
        background-color: {card_bg};
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
        margin: 20px auto;
        max-width: 95%;
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: black;
        font-weight: bold;
        border-radius: 12px;
        transition: 0.3s;
        border: none;
        padding: 8px 16px;
    }}
    .stButton>button:hover {{
        background-color: #8CD600;
        color: white;
    }}
    footer {{
        text-align: center;
        color: #aaa;
        font-size: 12px;
        margin-top: 30px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- HEADER -----------------
st.markdown(
    """
    <div class="logo" style="display:flex;justify-content:center;align-items:center;">
        <img src="https://upload.wikimedia.org/wikipedia/sco/2/21/Nvidia_logo.svg" height="70" style="margin-right:12px;">
        <h1>NVIDIA Stock Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

ticker = "NVDA"
st.sidebar.success(f"📈 Tracking Ticker: **{ticker}**")

# ----------------- SIDEBAR -----------------
st.sidebar.header("⚙️ Settings")
time_steps = st.sidebar.number_input("Time steps (sequence length)", min_value=10, max_value=200, value=60)
future_days = st.sidebar.number_input("Days to predict (business days)", min_value=1, max_value=150, value=100)

# ----------------- HISTORICAL PREDICTIONS -----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Historical Predictions")

if st.button("Show Historical vs Predicted"):
    try:
        dates, actual, predicted = get_test_predictions(ticker, time_steps=time_steps)
    except FileNotFoundError as e:
        st.error(f"{e}\n⚠️ Please run data_prep_analysis.py and train the model first.")
    else:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name="Actual Price",
                                 line=dict(color=primary_color, width=2)))
        fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name="Predicted Price",
                                 line=dict(color="cyan", dash="dash", width=2)))

        fig.update_layout(
            title=f"{ticker} — Historical vs Predicted Stock Prices",
            template="plotly_dark",
            paper_bgcolor=secondary_color,
            plot_bgcolor="#181818",
            font=dict(color="white"),
            xaxis=dict(title="Date", showgrid=False),
            yaxis=dict(title="Adjusted Close Price (USD)", showgrid=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=primary_color)
        )

        st.plotly_chart(fig, use_container_width=True)

        mse = ((actual - predicted) ** 2).mean()
        rmse = mse ** 0.5
        st.success(f"📉 Root Mean Square Error (RMSE): **{rmse:.4f}**")
st.markdown('</div>', unsafe_allow_html=True)

# ----------------- FUTURE PREDICTIONS -----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Future Predictions")

if st.button("Predict Future Prices"):
    try:
        model, scaler = load_saved(ticker)
    except FileNotFoundError as e:
        st.error(f"{e}\n⚠️ Please run data_prep_analysis.py and train the model first.")
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
            if current.weekday() < 5:  # skip weekends
                future_dates.append(current)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines',
                                 name="Historical Price", line=dict(color=primary_color, width=2)))
        fig.add_trace(go.Scatter(x=pd.to_datetime(future_dates), y=future_pred.flatten(),
                                 mode='lines', name="Future Prediction",
                                 line=dict(color="cyan", dash="dash", width=2)))

        fig.update_layout(
            title=f"{ticker} — Actual vs Future Prediction",
            template="plotly_dark",
            paper_bgcolor=secondary_color,
            plot_bgcolor="#181818",
            font=dict(color="white"),
            xaxis=dict(title="Date", showgrid=False),
            yaxis=dict(title="Adjusted Close Price (USD)", showgrid=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=primary_color)
        )

        st.plotly_chart(fig, use_container_width=True)

        out_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred.flatten()})
        out_df["Date"] = pd.to_datetime(out_df["Date"]).dt.date
        st.subheader("📅 Predicted Prices (Table)")
        st.dataframe(out_df, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown(
    "<footer>⚠️ Predictions are experimental and should not be considered financial advice.</footer>",
    unsafe_allow_html=True
)
