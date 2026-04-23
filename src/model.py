import tensorflow as tf
import random
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Layer
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

SEQUENCE_LENGTH = 60
FEATURES = [
    "Close", "High", "Low", "Open", "Volume",
    "MA20", "MA50", "Daily_Return", "Volume_Delta",
    "RSI", "MACD", "MACD_Signal", "Intraday_Range",
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


class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(
            shape=(input_shape[1], 1), initializer="zeros", trainable=True)

    def call(self, x):
        score = K.tanh(K.dot(x, self.W) + self.b)       # (batch, timesteps, 1)
        weights = K.softmax(score, axis=1)                # attention weights
        context = x * weights                             # weighted sequences
        return K.sum(context, axis=1)                     # collapse timesteps


def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)   # ← return_sequences=True now
    x = Dropout(0.2)(x)
    x = AttentionLayer()(x)                  # ← attention collapses timesteps
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
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

    for ticker in ["IVV.AX", "VAS.AX", "NDQ.AX", "VGS.AX"]:
        print(f"\n{'='*50}")
        print(f"Training model for {ticker}...")
        print(f"{'='*50}")

        df = download_data(ticker, TRAIN_START, TRAIN_END)
        macro = load_macro_data(TRAIN_START, TRAIN_END)
        df = engineer_features(df, macro)

        X, y, scaler = prepare_sequences(df)
        X_train, X_val, y_train, y_val = split_sequences(X, y)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")

        model = build_model((X_train.shape[1], X_train.shape[2]))
        history = train_model(model, X_train, y_train, X_val, y_val)
        save_model(model, scaler, ticker)

        print(f"Done — {ticker} model saved!")
