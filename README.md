# 📈 NVIDIA Stock Predictor + News Sentiment Tracker

Predict NVIDIA (NVDA) stock prices using an LSTM model and see latest NVIDIA news with sentiment analysis.

## 🚀 Features
- Fetch NVIDIA stock data (Yahoo Finance)
- LSTM model for stock price prediction
- Latest NVIDIA news (NewsAPI)
- Sentiment analysis (Positive / Negative / Neutral)

## 🛠 Setup
```bash
python -m pip install -r requirements.txt
python -m textblob.download_corpora
```

Get a free API key from [https://newsapi.org](https://newsapi.org) and put it in `app.py`.

## 📊 Usage
```bash
python data_prep_analysis.py --ticker NVDA --start 2015-01-01 --end 2025-01-01
python lstm_model.py --ticker NVDA --epochs 15
python -m streamlit run app.py
```
