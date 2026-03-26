import pandas as pd
import numpy as np


def add_moving_averages(df):
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    return df


def add_daily_return(df):
    df["Daily_Return"] = df["Close"].pct_change()
    return df


def add_volume_delta(df):
    df["Volume_Delta"] = df["Volume"].pct_change()
    return df


def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def engineer_features(df, macro_df=None):
    df = df.copy()
    df = add_moving_averages(df)
    df = add_daily_return(df)
    df = add_volume_delta(df)
    df = add_rsi(df)
    df = add_macd(df)

    if macro_df is not None:
        df = df.join(macro_df, how="left")
        df.ffill(inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    from data_loader import download_data, load_macro_data
    from data_loader import TRAIN_START, TRAIN_END

    print("Loading IVV.AX data...")
    df = download_data("IVV.AX", TRAIN_START, TRAIN_END)

    print("Loading macro data...")
    macro = load_macro_data(TRAIN_START, TRAIN_END)

    print("Engineering features...")
    featured = engineer_features(df, macro)

    print(featured.head())
    print(f"\nShape: {featured.shape}")
    print(f"\nColumns: {list(featured.columns)}")
