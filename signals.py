import numpy as np

def generate_signals(df):
    buy, sell = [], []

    for i in range(len(df)):
        if i == 0:
            buy.append(np.nan)
            sell.append(np.nan)
            continue

        if df["MA20"].iloc[i] > df["MA50"].iloc[i] and df["MA20"].iloc[i-1] <= df["MA50"].iloc[i-1]:
            buy.append(df["Close"].iloc[i])
            sell.append(np.nan)

        elif df["MA20"].iloc[i] < df["MA50"].iloc[i] and df["MA20"].iloc[i-1] >= df["MA50"].iloc[i-1]:
            sell.append(df["Close"].iloc[i])
            buy.append(np.nan)
        else:
            buy.append(np.nan)
            sell.append(np.nan)

    return buy, sell
