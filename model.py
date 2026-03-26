import joblib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


SEQUENCE_LENGTH = 60
FEATURES = [
    "Close", "High", "Low", "Open", "Volume",
    "MA20", "MA50", "Daily_Return", "Volume_Delta",
    "RSI", "MACD", "MACD_Signal",
    "^VIX", "^TNX", "GLD", "^GSPC", "AUDUSD=X"
]
TARGET = "Close"
MODEL_DIR = "models"


def prepare_sequences(df, sequence_length=SEQUENCE_LENGTH):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    X, y = [], []
    close_idx = FEATURES.index(TARGET)
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i])
        y.append(scaled[i, close_idx])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler


def split_sequences(X, y, val_split=0.1):
    split = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    return X_train, X_val, y_train, y_val


def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    return history


def save_model(model, scaler, ticker):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/{ticker}_model.keras")
    joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_scaler.pkl")
    print(f"Model and scaler saved for {ticker}")


if __name__ == "__main__":
    from data_loader import download_data, load_macro_data
    from data_loader import TRAIN_START, TRAIN_END
    from features import engineer_features

    ticker = "IVV.AX"
    print(f"Loading data for {ticker}...")
    df = download_data(ticker, TRAIN_START, TRAIN_END)
    macro = load_macro_data(TRAIN_START, TRAIN_END)
    df = engineer_features(df, macro)

    print("Preparing sequences...")
    X, y, scaler = prepare_sequences(df)
    X_train, X_val, y_train, y_val = split_sequences(X, y)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    print("Building model...")
    model = build_model((X_train.shape[1], X_train.shape[2]))

    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    save_model(model, scaler, ticker)
    print("Done!")
