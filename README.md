# NIFTY Options Strategy Backtester

This project backtests a simple options trading strategy on NIFTY using RSI, MACD, and SMA signals on 15-minute candles.

- Underlying: NIFTY (^NSEI from Yahoo Finance)
- Signals:
  - Buy CALL when RSI<30, Close>SMA20, MACD>Signal
  - Buy PUT when RSI>70, Close<SMA20, MACD<Signal
- Execution: Buy 1 ATM weekly option (approx priced with Black–Scholes using realized vol proxy)
- Risk: Stop-loss 30%, Take-profit 60%, end-of-day/expiry exit, max 2 days holding

## Quick start

1) Create/activate a venv and install deps:

```
# optional if not using existing venv
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```

2) Run the backtest:

```
python main.py
```

3) Fetch NSE option chain with expiry filter:

```
# Indices (NIFTY/BANKNIFTY), list expiries and preview
python main.py option-chain NIFTY

# Filter by expiry (format must match NSE, e.g., 21-Aug-2025)
python main.py option-chain NIFTY --expiry 21-Aug-2025

# Save to CSV
python main.py option-chain NIFTY --expiry 21-Aug-2025 --csv nifty_chain.csv

# Equities endpoint example
python main.py option-chain RELIANCE --equity --expiry 28-Aug-2025 --csv rel_chain.csv
```

4) Get a trade recommendation from option-chain data:

```
# For indices
python main.py signal NIFTY
python main.py signal NIFTY --expiry 21-Aug-2025

# For an equity
python main.py signal RELIANCE --equity --expiry 28-Aug-2025 --sl 0.25 --tp 0.5
```

5) Run the Streamlit UI (symbol/expiry selection + recommendation):

```
streamlit run streamlit_app.py
```

## Notes
- Educational only; not trading advice.
- For live trading, replace pricing with real option chain quotes and integrate a broker API.
