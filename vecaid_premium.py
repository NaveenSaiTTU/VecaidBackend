import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import pywt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib

FORECAST_HORIZON = 2

def safe_float(val):
    if np.isscalar(val):
        return float(val)
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return float(val.item())
        else:
            return float(val[0])
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    raise ValueError("Cannot convert input to float.")

def get_sentiment(ticker):
    try:
        t = yf.Ticker(ticker)
        news = t.news
        if not news or len(news) == 0:
            return 0.0
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(article.get('title', ''))['compound'] for article in news if article.get('title', '')]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamentals = {
        "trailingPE": info.get("trailingPE", np.nan),
        "dividendYield": info.get("dividendYield", np.nan),
        "priceToBook": info.get("priceToBook", np.nan),
        "profitMargins": info.get("profitMargins", np.nan)
    }
    return {k: np.nan_to_num(v, nan=0) for k, v in fundamentals.items()}

def wavelet_denoise(series, wavelet='db4', level=1):
    coeff = pywt.wavedec(series, wavelet, mode="per")
    coeff[1:] = [np.zeros_like(detail) for detail in coeff[1:]]
    return pywt.waverec(coeff, wavelet, mode="per")[:len(series)]

def create_lag_features(df, n_lags=5):
    df = df.copy()
    df['Close_denoised'] = wavelet_denoise(df['Close'])
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['Close_denoised'].shift(i)
    df.dropna(inplace=True)
    return df

def enhanced_lstm_forecast(df):
    data = df['Close'].values.ravel()
    window = 10
    if len(data) < window:
        return float(data[-1])
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    X = np.array(X).reshape((len(X), window, 1))
    y = np.array(y)
    model = models.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(64, activation='relu', return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    last_seq = data[-window:].reshape((1, window, 1))
    pred = model.predict(last_seq, verbose=0)
    return float(pred[0, 0])

def svr_forecast(df):
    data = df['Close'].values.ravel()
    window = 10
    if len(data) <= window:
        return float(data[-1])
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_scaled, y)
    last_seq = data[-window:]
    last_seq_scaled = scaler.transform(last_seq.reshape(1, -1))
    pred = model.predict(last_seq_scaled)
    return float(pred[0])

def final_combiner_model(df, base_model, options_strike, fundamentals, ticker, linreg_length=14):
    df = df.copy()
    df["Close_denoised"] = wavelet_denoise(df["Close"])
    df["target"] = df["Close"].shift(-FORECAST_HORIZON)
    df["sma_200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["linreg_14"] = df["Close"].rolling(linreg_length, min_periods=1).apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0], raw=False)
    df["sentiment_score"] = get_sentiment(ticker)
    df = df[df["target"].notna()].copy()
    df.reset_index(drop=True, inplace=True)

    df["opt_strike"] = options_strike or 0.0
    for k, v in fundamentals.items():
        df[k] = v

    tech_cols = ["sma_200", "linreg_14", "sentiment_score"]
    df[tech_cols] = df[tech_cols].fillna(0).replace([np.inf, -np.inf], 0)
    xgb_cols = ["Adj Close", "High", "Low", "Volume", "Close_denoised"] + [f'lag_{i}' for i in range(1, 6)]

    df["base_pred"] = base_model.predict(df[xgb_cols])
    df["enhanced_lstm_pred"] = enhanced_lstm_forecast(df)
    df["svr_pred"] = svr_forecast(df)

    features = df[["base_pred", "enhanced_lstm_pred", "svr_pred", "opt_strike", "linreg_14", "sma_200", "sentiment_score"]]
    target = df["target"]
    features = features.fillna(0).replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    meta_model = GradientBoostingRegressor(n_estimators=150, max_depth=5)
    meta_model.fit(scaled, target)

    latest = features.iloc[[-1]]
    final_pred = float(meta_model.predict(scaler.transform(latest))[0])
    return final_pred

def predict_forecast(df, options_strike, fundamentals, ticker, min_train_size=50):
    df = create_lag_features(df)
    df["target"] = df["Close"].shift(-FORECAST_HORIZON)
    df.dropna(inplace=True)

    if len(df) < min_train_size:
        return None

    model_path = "xgb_model.pkl"
    if not os.path.exists(model_path):
        return None

    base_model = joblib.load(model_path)
    return final_combiner_model(df, base_model, options_strike, fundamentals, ticker)

def get_fundamentals(ticker):
    return {"trailingPE": 20, "dividendYield": 0.01, "priceToBook": 3, "profitMargins": 0.2}

def options_contracts_signal(ticker, std_multiplier=2):
    return "calls", 1000000, 170.0  # dummy

if __name__ == "__main__":
    ticker = "AAPL"
    data = yf.download(ticker, start="2021-01-01", end=datetime.date.today().strftime('%Y-%m-%d'))[['Adj Close','Close','High','Low','Volume']].copy()
    data = create_lag_features(data)
    data["target"] = data["Close"].shift(-FORECAST_HORIZON)
    data.dropna(inplace=True)

    X_train = data.drop(["Close", "target"], axis=1)
    y_train = data["target"]
    model = XGBRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "xgb_model.pkl")
    print("✅ Model saved as xgb_model.pkl")