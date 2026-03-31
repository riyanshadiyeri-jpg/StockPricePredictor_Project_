import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from data_loader import TEST_START, TEST_END
from model import FEATURES, SEQUENCE_LENGTH, MODEL_DIR
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_saved_model(ticker):
    model_path = f"{MODEL_DIR}/{ticker}_model.keras"
    scaler_path = f"{MODEL_DIR}/{ticker}_scaler.pkl"

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model and scaler loaded for {ticker}")
    return model, scaler


def prepare_test_data(ticker, model, scaler):
    from data_loader import download_data, load_macro_data
    from features import engineer_features

    print(f"Downloading 2025 test data for {ticker}...")
    df = download_data(ticker, TEST_START, TEST_END)
    macro = load_macro_data(TEST_START, TEST_END)
    df = engineer_features(df, macro)

    scaled = scaler.transform(df[FEATURES])

    X_test, y_test = [], []
    close_idx = FEATURES.index("Close")

    for i in range(SEQUENCE_LENGTH, len(scaled)):
        X_test.append(scaled[i-SEQUENCE_LENGTH:i])
        y_test.append(scaled[i, close_idx])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_test, y_test, df


def make_predictions(model, scaler, X_test):
    scaled_preds = model.predict(X_test)

    dummy = np.zeros((len(scaled_preds), len(FEATURES)))
    close_idx = FEATURES.index("Close")
    dummy[:, close_idx] = scaled_preds[:, 0]

    predictions = scaler.inverse_transform(dummy)[:, close_idx]
    return predictions


def calculate_metrics(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return mae, rmse, mape


def plot_predictions(df, predictions, actual, ticker):
    dates = df.index[SEQUENCE_LENGTH:]

    plt.figure(figsize=(14, 6))
    plt.plot(dates, actual, label="Actual Price", color="blue", linewidth=1.5)
    plt.plot(dates, predictions, label="Predicted Price",
             color="orange", linewidth=1.5, linestyle="--")
    plt.title(f"{ticker} — Predicted vs Actual Closing Price (2025)")
    plt.xlabel("Date")
    plt.ylabel("Price (AUD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{ticker}_prediction.png", dpi=150)
    plt.show()
    print(f"Plot saved to plots/{ticker}_prediction.png")


def evaluate_ticker(ticker):
    model, scaler = load_saved_model(ticker)
    X_test, y_test, df = prepare_test_data(ticker, model, scaler)

    print(f"\nMaking predictions for {ticker}...")
    predictions = make_predictions(model, scaler, X_test)

    close_idx = FEATURES.index("Close")
    dummy_actual = np.zeros((len(y_test), len(FEATURES)))
    dummy_actual[:, close_idx] = y_test
    actual = scaler.inverse_transform(dummy_actual)[:, close_idx]

    print(f"\nAccuracy metrics for {ticker}:")
    mae, rmse, mape = calculate_metrics(actual, predictions)

    plot_predictions(df, predictions, actual, ticker)
    return mae, rmse, mape


if __name__ == "__main__":
    results = {}

    for ticker in ["IVV.AX", "VAS.AX", "NDQ.AX", "VGS.AX"]:
        print(f"\n{'='*50}")
        print(f"Evaluating {ticker}...")
        print(f"{'='*50}")
        mae, rmse, mape = evaluate_ticker(ticker)
        results[ticker] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    print(f"\n{'='*50}")
    print("PORTFOLIO SUMMARY")
    print(f"{'='*50}")
    print(f"{'Ticker':<10} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("-" * 38)
    for ticker, metrics in results.items():
        print(
            f"{ticker:<10} ${metrics['MAE']:>6.2f} ${metrics['RMSE']:>6.2f} {metrics['MAPE']:>6.2f}%")
