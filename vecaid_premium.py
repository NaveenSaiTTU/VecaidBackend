import os
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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import math
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

FORECAST_HORIZON = 2

def safe_float(val):
    if np.isscalar(val):
        return float(val)
    if isinstance(val, np.ndarray):
        return float(val.item()) if val.size == 1 else float(val[0])
    if hasattr(val, 'iloc'):
        return float(val.iloc[0])
    raise ValueError("Cannot convert input to float.")

def safe_polyfit(x, y, deg=1):
    try:
        coeff = np.polyfit(x, y, deg)
        return coeff[0]
    except Exception:
        return np.nan

def get_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news
        if not news:
            return 0.0
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(article.get('title', ''))['compound'] for article in news if article.get('title')]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

def get_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {k: np.nan_to_num(info.get(k, np.nan), nan=0) for k in ["trailingPE", "dividendYield", "priceToBook", "profitMargins"]}

def markov_chain_analysis(ticker, start_date="2021-01-01", end_date=datetime.date.today().strftime('%Y-%m-%d')):
    data = yf.download(ticker, start=start_date, end=end_date)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    data["daily_return"] = data[price_col].pct_change()
    data["state"] = np.where(data["daily_return"] >= 0, "up", "down")
    num = len(data[(data["state"]=="up") & (data["state"].shift(-1)=="down") & (data["state"].shift(-5)=="down")])
    den = len(data[(data["state"].shift(1)=="down") & (data["state"].shift(5)=="down")])
    return num / den if den != 0 else 0.0

def compute_linreg(df):
    return df.rolling(14, min_periods=1).apply(lambda s: safe_polyfit(np.arange(len(s)), s, 1), raw=False)

def final_combiner_model(df, base_model, options_strike, fundamentals, ticker, linreg_length=14):
    df = df.copy()
    df["linreg_14"] = compute_linreg(df["Close"])
    df["markov_prob"] = markov_chain_analysis(ticker)
    df["sentiment_score"] = get_sentiment(ticker)

    df["opt_strike"] = options_strike if options_strike is not None else 0.0
    df.update(pd.DataFrame([fundamentals]))

    if "target" not in df.columns:
        df["target"] = df["Close"].shift(-FORECAST_HORIZON)
    df.dropna(subset=["target"], inplace=True)

    tech_features = ["linreg_14", "markov_prob", "sentiment_score"]
    df[tech_features] = df[tech_features].ffill().fillna(df[tech_features].median()).fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0)

    if df.empty:
        print("No data available after feature generation.")
        return None

    X = df[tech_features]
    y = df["target"]

    if X.empty or y.empty:
        print("No samples available for training.")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GradientBoostingRegressor()
    model.fit(X_scaled, y)

    latest = df.iloc[-1:][tech_features]
    pred = model.predict(scaler.transform(latest))
    return float(pred[0])

def predict_forecast(df_train, options_strike, fundamentals, ticker, min_train_size=50):
    df_train = df_train.copy()
    df_train["target"] = df_train["Close"].shift(-FORECAST_HORIZON)
    df_train.dropna(subset=["target"], inplace=True)
    if len(df_train) < min_train_size:
        return None

    train_size = int(len(df_train) * 0.8)
    train_df = df_train.iloc[:train_size]

    X_train = train_df.drop(columns=["Close", "target"], errors='ignore')
    y_train = train_df["target"] if "target" in train_df else pd.Series([0]*len(train_df))

    if X_train.empty or y_train.empty:
        print("Training data is empty after filtering.")
        return None

    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train.values)

    xgb = XGBRegressor()
    param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(scaled_X_train, y_train)
    base_model = grid_search.best_estimator_

    return final_combiner_model(df_train, base_model, options_strike, fundamentals, ticker, linreg_length=14)

if __name__ == "__main__":
    ticker = "AAPL"
    stock_data = yf.download(ticker, start="2021-01-01", end=datetime.date.today().strftime('%Y-%m-%d'), auto_adjust=False)[['Close','High','Low','Volume']].copy()
    options_strike = 150.0  # mock
    fundamentals = get_fundamentals(ticker)
    pred = predict_forecast(stock_data, options_strike, fundamentals, ticker)
    if pred is not None:
        print(f"Predicted price for {ticker}: {pred}")
        with open("xgb_model.pkl", "wb") as f:
            import pickle
            pickle.dump(pred, f)
        print("✅ Model saved as xgb_model.pkl")
    else:
        print("❌ Prediction failed due to insufficient or invalid data.")