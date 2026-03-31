from model import FEATURES, SEQUENCE_LENGTH, MODEL_DIR
from features import engineer_features
from data_loader import download_data, load_macro_data, TEST_START, TEST_END
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


st.set_page_config(
    page_title="ASX ETF Price Predictor",
    page_icon="📈",
    layout="wide"
)


@st.cache_resource
def load_cached_model(ticker):
    model_path = f"{MODEL_DIR}/{ticker}_model.keras"
    scaler_path = f"{MODEL_DIR}/{ticker}_scaler.pkl"
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


@st.cache_data
def get_predictions(ticker):
    model, scaler = load_cached_model(ticker)

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

    scaled_preds = model.predict(X_test, verbose=0)

    dummy_pred = np.zeros((len(scaled_preds), len(FEATURES)))
    dummy_pred[:, close_idx] = scaled_preds[:, 0]
    predictions = scaler.inverse_transform(dummy_pred)[:, close_idx]

    dummy_actual = np.zeros((len(y_test), len(FEATURES)))
    dummy_actual[:, close_idx] = y_test
    actual = scaler.inverse_transform(dummy_actual)[:, close_idx]

    dates = df.index[SEQUENCE_LENGTH:]

    mae = np.mean(np.abs(actual - predictions))
    rmse = np.sqrt(np.mean((actual - predictions) ** 2))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    return predictions, actual, dates, mae, rmse, mape


st.sidebar.title("📈 ETF Predictor")
st.sidebar.markdown("---")

ticker = st.sidebar.selectbox(
    "Select ETF",
    ["IVV.AX", "VAS.AX", "NDQ.AX", "VGS.AX"],
    help="Choose which ASX ETF to analyse"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This dashboard uses an LSTM neural network trained on 
2022–2024 market data to predict 2025 closing prices.

**Features used:**
- OHLCV price data
- Technical indicators (RSI, MACD, MA20, MA50)
- Macro signals (VIX, Treasury yields, Gold, S&P500, AUD/USD)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Riyansh Adiyeri")
st.sidebar.markdown("Macquarie University — BCom/IT")

st.title(f"ASX ETF Price Predictor — {ticker}")
st.markdown("LSTM neural network predictions vs real 2025 market prices")
st.markdown("---")

with st.spinner(f"Loading predictions for {ticker}..."):
    predictions, actual, dates, mae, rmse, mape = get_predictions(ticker)

st.subheader("Model Accuracy — 2025 Test Period")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="MAPE",
        value=f"{mape:.2f}%",
        help="Mean Absolute Percentage Error — average % difference between predicted and actual"
    )

with col2:
    st.metric(
        label="MAE",
        value=f"${mae:.2f}",
        help="Mean Absolute Error — average dollar difference between predicted and actual"
    )

with col3:
    st.metric(
        label="RMSE",
        value=f"${rmse:.2f}",
        help="Root Mean Squared Error — penalises large errors more heavily than MAE"
    )

st.markdown("---")
st.subheader("Predicted vs Actual Closing Price (2025)")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, actual, label="Actual Price", color="#1f77b4", linewidth=1.5)
ax.plot(dates, predictions, label="Predicted Price",
        color="#ff7f0e", linewidth=1.5, linestyle="--")
ax.set_xlabel("Date")
ax.set_ylabel("Price (AUD)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

st.pyplot(fig)
plt.close()

st.markdown("---")
st.subheader("Portfolio Overview — All 4 ETFs")

portfolio_data = []
for t in ["IVV.AX", "VAS.AX", "NDQ.AX", "VGS.AX"]:
    try:
        with st.spinner(f"Loading {t}..."):
            _, _, _, t_mae, t_rmse, t_mape = get_predictions(t)
        portfolio_data.append({
            "Ticker": t,
            "MAPE (%)": round(t_mape, 2),
            "MAE ($)": round(t_mae, 2),
            "RMSE ($)": round(t_rmse, 2)
        })
    except Exception as e:
        st.warning(f"Could not load {t}: {e}")

if portfolio_data:
    portfolio_df = pd.DataFrame(portfolio_data)
    st.dataframe(
        portfolio_df.style.highlight_min(
            subset=["MAPE (%)"], color="lightgreen"
        ).highlight_max(
            subset=["MAPE (%)"], color="#ffcccc"
        ),
        use_container_width=True
    )

st.markdown("---")
st.subheader("Raw Prediction Data")

if st.checkbox("Show raw data table"):
    raw_df = pd.DataFrame({
        "Date": dates,
        "Actual Price (AUD)": actual.round(2),
        "Predicted Price (AUD)": predictions.round(2),
        "Error ($)": (actual - predictions).round(2),
        "Error (%)": ((actual - predictions) / actual * 100).round(2)
    })
    raw_df = raw_df.set_index("Date")
    st.dataframe(raw_df, use_container_width=True)

st.markdown("---")
st.caption("Data sourced from Yahoo Finance via yfinance. "
           "Trained on ASX ETF data from January 2022 to December 2024. "
           "Predictions cover January 2025 to December 2025.")
