# Stock Price Predictor

A machine learning project I built after getting into investing and index funds. The model predicts the closing prices of four ASX-listed ETFs using a Bidirectional LSTM neural network with an attention layer, technical indicators, and macroeconomic factors. Trained on post-COVID market data (January 2022 to December 2024) and tested against real 2025 prices.

## Project Structure

```
StockPricePredictor_Project/
│
├── src/
│   ├── data_loader.py       # Downloads stock + macro data from Yahoo Finance
│   ├── features.py          # Engineers technical indicators from raw price data
│   ├── model.py             # Builds and trains the LSTM neural network
│   └── evaluate.py          # Measures prediction accuracy against real 2025 prices
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_and_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── data/
│   ├── raw/                 # Downloaded CSVs
│   └── processed/           # Feature engineered data
│
├── models/                  # Saved .keras models and scalers
├── plots/                   # Prediction charts
├── requirements.txt
└── README.md
```

## ETFs Analysed

I picked these four because they are the ones I personally invest in or have been looking at. All four are broad market index funds.

| Ticker | Name | What it tracks |
|--------|------|----------------|
| IVV.AX | iShares S&P 500 ETF | Top 500 US companies, AUD denominated |
| VAS.AX | Vanguard Australian Shares | Top 300 ASX companies |
| VGS.AX | Vanguard Global Shares | Global developed markets ex-Australia |
| NDQ.AX | BetaShares Nasdaq 100 | Top 100 US tech stocks, AUD denominated |

## Macroeconomic Features

These were chosen because they directly influence the kind of broad market ETFs I am predicting.

| Ticker | What it is | Why it matters |
|--------|-----------|----------------|
| ^VIX | CBOE Volatility Index | Market fear gauge, spikes when investors are uncertain |
| ^TNX | 10-Year US Treasury Yield | Rising yields historically pressure equity prices |
| GLD | Gold ETF price | Safe-haven flows tend to move inverse to equities |
| ^GSPC | S&P 500 Index | Direct benchmark for IVV.AX and NDQ.AX |
| AUDUSD=X | AUD/USD Exchange Rate | Currency movements directly affect AUD-denominated ETFs |

## Data

- **Source:** Yahoo Finance via the yfinance Python library
- **Training:** January 2022 to December 2024
- **Test:** January 2025 to December 2025 (data the model has never seen)
- **Raw features:** Open, High, Low, Close, Volume (OHLCV)

### Engineered Features

These are all calculated from raw OHLCV data before being fed into the model. Raw closing prices alone are not enough for the LSTM to learn from because they do not encode any information about momentum, trend, or trading activity.

| Feature | What it measures |
|---------|-----------------|
| MA20 | 20-day moving average, short-term price trend |
| MA50 | 50-day moving average, longer-term price trend |
| Daily Return | Percentage price change from the previous day |
| Volume Delta | Percentage change in trading volume from the previous day |
| RSI | Relative Strength Index, overbought/oversold signal (0 to 100) |
| MACD | Moving Average Convergence Divergence, trend momentum signal |
| MACD Signal | 9-day average of MACD, used to detect momentum shifts |
| Intraday Range | Daily high minus daily low, measures daily volatility |

## Model Architecture

### Why LSTM?

A regular neural network treats each day as completely independent with no memory of what came before. LSTM (Long Short-Term Memory) was specifically designed for sequences, so it can learn patterns that play out over days, weeks, and months. For example, identifying when a 20-day moving average crosses above a 50-day moving average requires the model to remember prices from months ago, which a regular network cannot do.

### Why Bidirectional?

A standard LSTM only reads the sequence forward. A Bidirectional LSTM reads it both forwards and backwards, which gives it more context about each sequence and generally improves accuracy on time series tasks.

### Why an Attention Layer?

The attention layer lets the model learn which timesteps in the 60-day input window are most important for predicting the next price. Instead of treating all 60 days equally, it assigns higher weights to the days that matter most. This was added after the initial LSTM version and improved results.

### Architecture Summary

```
Input: (60 timesteps, 18 features)
    Bidirectional LSTM (128 units, return_sequences=True)
    Dropout (20%)
    LSTM (64 units, return_sequences=True)
    Dropout (20%)
    Attention Layer (custom)
    Dense (32 units, ReLU)
    Dense (1 unit) -> predicted Close price
```

### Overfitting Prevention

- **Dropout (20%):** randomly disables 20% of neurons each training step so the model cannot just memorise sequences
- **Chronological split:** 90% training, last 10% validation, never shuffled. Shuffling would leak future data into training which would make the accuracy metrics meaningless

### Known Limitations

- Does not capture event-driven shocks like Fed announcements or geopolitical events
- Monthly macro data like CPI and unemployment was not included due to date alignment complexity
- No model can reliably predict stock prices. This project is about building a rigorous ML pipeline, not a trading strategy

## How to Run

1. Clone the repo and set up a virtual environment

```bash
git clone <repo-url>
cd StockPricePredictor_Project
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Run the notebooks in order from the `notebooks/` folder

```
01_data_collection.ipynb       # downloads and saves raw data
02_eda_and_preprocessing.ipynb # feature engineering and EDA
03_model_training.ipynb        # trains and saves models (30-60 min on CPU)
04_evaluation.ipynb            # loads models and evaluates on 2025 data
```

## Results

All four models were evaluated on real 2025 market data they had never seen during training.

| Ticker | MAE | RMSE | MAPE | Directional Accuracy |
|--------|-----|------|------|----------------------|
| IVV.AX | $7.07 | $7.20 | 10.64% | 51.77% |
| VAS.AX | $2.64 | $2.99 | 2.47% | 41.84% |
| NDQ.AX | $2.74 | $2.98 | 4.96% | 49.65% |
| VGS.AX | $14.04 | $14.27 | 9.46% | 56.03% |
| **Portfolio avg** | | | **6.88%** | **49.82%** |

### Key Observations

VAS.AX was the strongest performer at 2.47% MAPE, meaning the model's predictions were on average within $2.64 of the real price across 2025. VGS.AX had the highest dollar error ($14.04 MAE) which makes sense given it trades at a much higher price than the other three ETFs, so the same percentage error produces a larger dollar figure.

Directional accuracy was mixed. VGS.AX (56.03%) and IVV.AX (51.77%) both beat the 50% random baseline, meaning the model correctly predicted whether the price would go up or down on more days than not. VAS.AX (41.84%) and NDQ.AX (49.65%) came in below 50%, which suggests the model captures the broader price trend well but struggles with day-to-day direction on those tickers.

The consistent gap between predicted and actual prices across all four ETFs comes down to the training window. The model learned from 2022 to 2024, which included significant volatility and drawdowns, so it tends to underestimate the magnitude of the sustained 2025 bull market. This is a training window limitation, not a flaw in the architecture.

### Prediction Plots

![IVV.AX](plots/IVV.AX_prediction.png)
![VAS.AX](plots/VAS.AX_prediction.png)
![NDQ.AX](plots/NDQ.AX_prediction.png)
![VGS.AX](plots/VGS.AX_prediction.png)

## Future Improvements

- News sentiment pipeline using FinBERT to capture event-driven price movements
- Walk-forward validation for more rigorous out-of-sample testing
- Transformer architecture to replace LSTM
- Extending training data back to 2015 to cover more market regimes

## Author

Riyansh Ritesh Adiyeri
Macquarie University, Bachelor of Commerce / Information Technology

Major: International Business and Cyber Security

[GitHub](https://github.com/riyanshadiyeri-jpg/StockPricePredictor_Project_) | [LinkedIn](https://www.linkedin.com/in/riyansh-adiyeri-46a9852a6/?skipRedirect=true)
