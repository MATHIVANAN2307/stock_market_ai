import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker, start="2012-01-01"):
    df = yf.download(ticker, start=start)

    # Handle MultiIndex columns (Yahoo Finance issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    return df


def add_moving_averages(df):
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df


def support_resistance(df, window=20):
    support = df["Low"].rolling(window).min()
    resistance = df["High"].rolling(window).max()
    return support, resistance


def prepare_data(df, seq_len=100):
    scaler = MinMaxScaler()
    close_prices = df[["Close"]]
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler
