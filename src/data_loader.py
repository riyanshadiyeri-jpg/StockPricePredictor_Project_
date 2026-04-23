import yfinance as yf
import pandas as pd

TICKERS = ["IVV.AX", "VAS.AX", "NDQ.AX", "VGS.AX"]

MACRO_TICKERS = ["^VIX", "^TNX", "GLD", "^GSPC", "AUDUSD=X"]

TRAIN_START = "2022-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"


def download_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.dropna(inplace=True)
    return df


def load_macro_data(start, end):
    macro_frames = []

    for ticker in MACRO_TICKERS:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        close = df[["Close"]].rename(columns={"Close": ticker})
        macro_frames.append(close)

    macro = pd.concat(macro_frames, axis=1)
    macro.dropna(inplace=True)
    return macro


if __name__ == "__main__":
    print("Downloading IVV.AX training data...")
    df = download_data("IVV.AX", TRAIN_START, TRAIN_END)
    print(df.head())
    print(f"\nShape: {df.shape}")

    print("\nDownloading macro data...")
    macro = load_macro_data(TRAIN_START, TRAIN_END)
    print(macro.head())
    print(f"\nMacro shape: {macro.shape}")
