import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
import joblib
from data_loader import download_data, load_macro_data, TEST_START, TEST_END
from features import engineer_features
SEQUENCE_LENGTH = 60
MODEL_DIR = "models"
FEATURES = [
    "Close", "High", "Low", "Open", "Volume",
    "MA20", "MA50", "Daily_Return", "Volume_Delta",
    "RSI", "MACD", "MACD_Signal", "Intraday_Range",
    "^VIX", "^TNX", "GLD", "^GSPC", "AUDUSD=X"
]
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


st.set_page_config(
    page_title="ASX ETF Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


st.set_page_config(
    page_title="ASX ETF Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2d3150;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4ade80;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 2px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e5e7eb;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3150;
        margin-bottom: 16px;
    }
    .feature-pill {
        display: inline-block;
        background: #1e2130;
        border: 1px solid #2d3150;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: #9ca3af;
        margin: 3px;
        cursor: help;
    }
    .ticker-badge {
        background: #1d4ed8;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stSelectbox label { color: #9ca3af !important; }
    div[data-testid="stSidebarContent"] {
        background-color: #1e2130;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_INFO = {
    "Close": ("Close Price", "The ETF's closing price for that trading day — the primary prediction target"),
    "High": ("Daily High", "The highest price reached during the trading day"),
    "Low": ("Daily Low", "The lowest price reached during the trading day"),
    "Open": ("Open Price", "The price the ETF opened at when the market opened"),
    "Volume": ("Volume", "Number of units traded that day — measures market activity"),
    "MA20": ("MA20", "20-day moving average — average closing price over the last 20 days, captures short term trends"),
    "MA50": ("MA50", "50-day moving average — average closing price over the last 50 days, captures longer term trends"),
    "Daily_Return": ("Daily Return", "Percentage price change from the previous day — normalises price movement"),
    "Volume_Delta": ("Volume Delta", "Percentage change in trading volume from the previous day — measures conviction behind price moves"),
    "RSI": ("RSI", "Relative Strength Index (0-100) — above 70 suggests overbought, below 30 suggests oversold"),
    "MACD": ("MACD", "Moving Average Convergence Divergence — difference between 12-day and 26-day EMAs, measures momentum"),
    "MACD_Signal": ("MACD Signal", "9-day EMA of MACD — when MACD crosses above this line it is a bullish signal"),
    "Intraday_Range": ("Intraday Range", "Daily high minus daily low — measures how volatile each trading day was"),
    "^VIX": ("VIX", "CBOE Volatility Index — the market fear gauge. Spikes during uncertainty and selloffs"),
    "^TNX": ("10Y Treasury Yield", "10-year US Treasury yield — rising yields historically pressure equity prices"),
    "GLD": ("Gold Price", "Gold ETF price — safe-haven flows often move inverse to broad equity ETFs"),
    "^GSPC": ("S&P 500", "S&P 500 index — direct benchmark that IVV.AX and NDQ.AX track"),
    "AUDUSD=X": ("AUD/USD Rate", "Australian dollar to US dollar exchange rate — directly affects AUD-denominated ETFs holding foreign assets"),
}

TICKER_INFO = {
    "IVV.AX": ("iShares S&P 500 ETF", "Tracks the top 500 US companies, traded in AUD on the ASX"),
    "VAS.AX": ("Vanguard Australian Shares", "Tracks the top 300 ASX-listed Australian companies"),
    "NDQ.AX": ("BetaShares Nasdaq 100", "Tracks the top 100 US technology stocks, traded in AUD"),
    "VGS.AX": ("Vanguard Global Shares", "Tracks global developed market equities excluding Australia"),
}

FINAL_RESULTS = {
    "IVV.AX": {"MAE": 3.26, "RMSE": 3.54, "MAPE": 4.85},
    "VAS.AX": {"MAE": 1.49, "RMSE": 1.84, "MAPE": 1.38},
    "NDQ.AX": {"MAE": 2.54, "RMSE": 2.75, "MAPE": 4.61},
    "VGS.AX": {"MAE": 10.70, "RMSE": 11.19, "MAPE": 7.15},
}


@st.cache_resource
def load_cached_model(ticker):
    model = load_model(f"{MODEL_DIR}/{ticker}_model.keras")
    scaler = joblib.load(f"{MODEL_DIR}/{ticker}_scaler.pkl")
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
    return predictions, actual, dates


with st.sidebar:
    st.markdown("## 📈 ASX ETF Predictor")
    st.markdown("---")

    ticker = st.selectbox(
        "Select ETF",
        list(TICKER_INFO.keys()),
        format_func=lambda x: f"{x} — {TICKER_INFO[x][0]}"
    )

    ticker_name, ticker_desc = TICKER_INFO[ticker]
    st.markdown(f"""
    <div style='background:#0f1117; border-radius:8px; padding:12px; 
                border:1px solid #2d3150; margin-top:8px;'>
        <div style='color:#e5e7eb; font-weight:600; font-size:0.9rem;'>{ticker_name}</div>
        <div style='color:#9ca3af; font-size:0.8rem; margin-top:4px;'>{ticker_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Info")
    st.markdown(f"""
    <div style='font-size:0.82rem; color:#9ca3af; line-height:1.8;'>
    🧠 &nbsp;<b style='color:#e5e7eb;'>Architecture</b>: 2-layer LSTM<br>
    📅 &nbsp;<b style='color:#e5e7eb;'>Training</b>: Jan 2022 – Dec 2024<br>
    🎯 &nbsp;<b style='color:#e5e7eb;'>Testing</b>: Jan 2025 – Dec 2025<br>
    🔁 &nbsp;<b style='color:#e5e7eb;'>Lookback</b>: 60 trading days<br>
    📊 &nbsp;<b style='color:#e5e7eb;'>Features</b>: 18 total<br>
    🌱 &nbsp;<b style='color:#e5e7eb;'>Random seed</b>: 42
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#6b7280;'>
    Built by <b style='color:#9ca3af;'>Riyansh Adiyeri</b><br>
    Macquarie University<br>
    BCom / IT — Data Science
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>
    <span class='ticker-badge'>{ticker}</span>
    <span style='font-size:1.5rem; font-weight:700; color:#e5e7eb;'>{ticker_name}</span>
</div>
<div style='color:#6b7280; font-size:0.9rem; margin-bottom:24px;'>{ticker_desc}</div>
""", unsafe_allow_html=True)

results = FINAL_RESULTS[ticker]
mape_color = "#4ade80" if results["MAPE"] < 4 else "#fbbf24" if results["MAPE"] < 7 else "#f87171"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:{mape_color};'>{results["MAPE"]:.2f}%</div>
        <div class='metric-label'>MAPE</div>
        <div class='metric-sub'>Mean Absolute % Error</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#60a5fa;'>${results["MAE"]:.2f}</div>
        <div class='metric-label'>MAE</div>
        <div class='metric-sub'>Mean Absolute Error</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#a78bfa;'>${results["RMSE"]:.2f}</div>
        <div class='metric-label'>RMSE</div>
        <div class='metric-sub'>Root Mean Squared Error</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_mape = np.mean([v["MAPE"] for v in FINAL_RESULTS.values()])
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#fb923c;'>{avg_mape:.2f}%</div>
        <div class='metric-label'>Portfolio Avg MAPE</div>
        <div class='metric-sub'>Across all 4 ETFs</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Predicted vs Actual Closing Price — 2025</div>",
            unsafe_allow_html=True)

with st.spinner(f"Loading predictions for {ticker}..."):
    predictions, actual, dates = get_predictions(ticker)

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

ax.plot(dates, actual, label="Actual Price",
        color="#60a5fa", linewidth=1.8, alpha=0.9)
ax.plot(dates, predictions, label="Predicted Price",
        color="#f97316", linewidth=1.8, linestyle="--", alpha=0.9)

ax.fill_between(dates, actual, predictions,
                alpha=0.08, color="#f97316")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=30, color="#9ca3af", fontsize=9)
plt.yticks(color="#9ca3af", fontsize=9)

ax.set_xlabel("Date", color="#6b7280", fontsize=10)
ax.set_ylabel("Price (AUD)", color="#6b7280", fontsize=10)
ax.legend(facecolor="#1e2130", edgecolor="#2d3150",
          labelcolor="#e5e7eb", fontsize=10)
ax.grid(True, alpha=0.1, color="#ffffff")

for spine in ax.spines.values():
    spine.set_edgecolor("#2d3150")

fig.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Features Used by the Model</div>",
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Technical Indicators**")
    tech_features = ["Close", "High", "Low", "Open", "Volume",
                     "MA20", "MA50", "Daily_Return", "Volume_Delta",
                     "RSI", "MACD", "MACD_Signal", "Intraday_Range"]
    pills_html = ""
    for f in tech_features:
        label, tooltip = FEATURE_INFO[f]
        pills_html += f"<span class='feature-pill' title='{tooltip}'>{label}</span>"
    st.markdown(pills_html, unsafe_allow_html=True)

with col_b:
    st.markdown("**Macroeconomic Signals**")
    macro_features = ["^VIX", "^TNX", "GLD", "^GSPC", "AUDUSD=X"]
    pills_html = ""
    for f in macro_features:
        label, tooltip = FEATURE_INFO[f]
        pills_html += f"<span class='feature-pill' title='{tooltip}'>{label}</span>"
    st.markdown(pills_html, unsafe_allow_html=True)

st.markdown("""
<div style='font-size:0.78rem; color:#6b7280; margin-top:8px;'>
💡 Hover over any feature pill to see what it measures
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>Portfolio Overview — All 4 ETFs</div>",
            unsafe_allow_html=True)

portfolio_rows = []
for t, metrics in FINAL_RESULTS.items():
    name, _ = TICKER_INFO[t]
    portfolio_rows.append({
        "Ticker": t,
        "Name": name,
        "MAPE (%)": metrics["MAPE"],
        "MAE ($)": metrics["MAE"],
        "RMSE ($)": metrics["RMSE"],
    })

portfolio_df = pd.DataFrame(portfolio_rows).set_index("Ticker")

st.dataframe(
    portfolio_df.style
    .highlight_min(subset=["MAPE (%)"], color="#14532d")
    .highlight_max(subset=["MAPE (%)"], color="#450a0a")
    .format({"MAPE (%)": "{:.2f}%", "MAE ($)": "${:.2f}", "RMSE ($)": "${:.2f}"}),
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)
if st.checkbox("Show raw prediction data"):
    raw_df = pd.DataFrame({
        "Date": dates,
        "Actual (AUD)": actual.round(2),
        "Predicted (AUD)": predictions.round(2),
        "Error ($)": (actual - predictions).round(2),
        "Error (%)": ((actual - predictions) / actual * 100).round(2)
    }).set_index("Date")
    st.dataframe(raw_df, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='font-size:0.75rem; color:#4b5563; text-align:center;'>
Data sourced from Yahoo Finance · Trained on ASX ETF data Jan 2022 – Dec 2024 · 
Predictions cover Jan 2025 – Dec 2025 · For educational purposes only
</div>
""", unsafe_allow_html=True)
