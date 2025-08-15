"""
Options strategy backtester for NIFTY using RSI + MACD + SMA.

What it does
- Downloads 15m candles for NIFTY (^NSEI) via yfinance
- Builds signals (buy CALL when RSI<30 & Close>SMA & MACD>Signal; buy PUT when RSI>70 & Close<SMA & MACD<Signal)
- Simulates buying 1 ATM weekly option (next Thursday expiry) on signal
- Prices options each bar with Black–Scholes using realized volatility as proxy
- Risk management: stop-loss %, take-profit %, time exit (end of session / expiry)
- Outputs trade log and performance summary

Notes
- This is an educational backtest, not investment advice.
- For live trading, integrate a broker API (e.g., Zerodha, AngelOne) and replace the pricing with real option chain quotes.
"""

from __future__ import annotations

import math
import argparse
import json
from typing import List, Optional
from dateutil import parser as dateparser
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
	import yfinance as yf
except Exception as e:  # pragma: no cover
	print("yfinance is required. Please install dependencies.")
	raise

try:
	import ta
except Exception as e:  # pragma: no cover
	print("ta is required. Please install dependencies.")
	raise

try:
	from nse_option_chain import NSEClient, list_expiry_dates, option_chain_to_df
except Exception:
	# Optional: user might only want to backtest
	NSEClient = None  # type: ignore
	list_expiry_dates = None  # type: ignore
	option_chain_to_df = None  # type: ignore
try:
	from strategies import analyze_option_chain
except Exception:
	analyze_option_chain = None  # type: ignore


# --- Constants ---
INDEX_TICKER = "^NSEI"  # NIFTY 50 Index on Yahoo
LOT_SIZE_NIFTY = 50
STRIKE_STEP = 50  # NIFTY strike increments
RISK_FREE_RATE = 0.06  # 6% annualized
IST_OFFSET = timedelta(hours=5, minutes=30)


def to_ist(ts: pd.Timestamp) -> pd.Timestamp:
	# yfinance returns tz-aware UTC; handle naive too
	if ts.tzinfo is None:
		return (ts + IST_OFFSET).tz_localize("Asia/Kolkata", nonexistent="shift_forward")
	return ts.tz_convert("Asia/Kolkata")


def next_thursday_expiry_ist(ts_ist: pd.Timestamp) -> pd.Timestamp:
	# Weekly expiry: Thursday 15:30 IST
	weekday = ts_ist.weekday()  # Monday=0 ... Sunday=6
	days_ahead = (3 - weekday) % 7  # 3 = Thursday
	expiry_date = (ts_ist + pd.Timedelta(days=days_ahead)).normalize() + pd.Timedelta(hours=15, minutes=30)
	# If already past expiry time on Thursday, go to next week
	if ts_ist > expiry_date or (weekday == 3 and ts_ist.time() >= pd.Timestamp("15:30").time()):
		expiry_date = expiry_date + pd.Timedelta(days=7)
	return expiry_date.tz_localize("Asia/Kolkata") if expiry_date.tzinfo is None else expiry_date


def round_to_nearest_strike(price: float, step: int = STRIKE_STEP) -> int:
	return int(round(price / step) * step)


def annualize_vol_from_intraday(returns: pd.Series, bars_per_day: int = 25, trading_days: int = 252) -> float:
	# returns should be log returns per bar
	sigma_bar = returns.std(ddof=0)
	return float(sigma_bar * math.sqrt(bars_per_day * trading_days))


def norm_cdf(x: float) -> float:
	# numerical stable CDF using erf
	return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
	S: float,
	K: float,
	T: float,
	r: float,
	sigma: float,
	option_type: str,
) -> float:
	"""Price a European option with Black–Scholes.

	S: underlying price
	K: strike
	T: time to expiry in years
	r: risk-free rate (annual)
	sigma: implied volatility (annual)
	option_type: 'C' or 'P'
	"""
	if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
		# Immediate expiry intrinsic value
		if option_type == "C":
			return max(0.0, S - K)
		else:
			return max(0.0, K - S)

	d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
	d2 = d1 - sigma * math.sqrt(T)
	if option_type == "C":
		return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
	else:
		return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


@dataclass
class Trade:
	direction: str  # 'CALL' or 'PUT'
	entry_time: pd.Timestamp
	exit_time: Optional[pd.Timestamp]
	strike: int
	expiry: pd.Timestamp
	qty_lots: int
	entry_price: float  # premium per unit
	exit_price: Optional[float] = None
	reason_exit: Optional[str] = None

	def pnl(self) -> Optional[float]:
		if self.exit_price is None:
			return None
		return (self.exit_price - self.entry_price) * self.qty_lots * LOT_SIZE_NIFTY


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
	close = df["Close"].copy()
	rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
	macd_i = ta.trend.MACD(close)
	macd = macd_i.macd()
	macd_signal = macd_i.macd_signal()
	sma = close.rolling(window=20, min_periods=20).mean()

	df = df.copy()
	df["rsi"] = rsi
	df["macd"] = macd
	df["macd_signal"] = macd_signal
	df["sma20"] = sma

	# Signals
	df["buy_call"] = (df["rsi"] < 30) & (close > df["sma20"]) & (df["macd"] > df["macd_signal"])
	df["buy_put"] = (df["rsi"] > 70) & (close < df["sma20"]) & (df["macd"] < df["macd_signal"])
	return df


def realized_vol_lookback(df: pd.DataFrame, lookback_bars: int = 100) -> float:
	# compute annualized realized vol from last N bars of 15m log returns
	log_ret = np.log(df["Close"]).diff().dropna()
	if len(log_ret) < max(20, lookback_bars // 2):
		return 0.18  # fallback
	return annualize_vol_from_intraday(log_ret.tail(lookback_bars))


def simulate_backtest(
	df: pd.DataFrame,
	sl_pct: float = 0.3,
	tp_pct: float = 0.6,
	max_holding_days: float = 2.0,
	lots: int = 1,
) -> Tuple[List[Trade], pd.DataFrame]:
	"""Simulate sequential trades on 15m bars.

	sl_pct: stop loss as fraction of premium (e.g., 0.3 = -30%)
	tp_pct: take profit as fraction of premium (e.g., 0.6 = +60%)
	max_holding_days: safety exit
	lots: number of lots per trade
	Returns: (trades, equity_curve_df)
	"""
	trades: List[Trade] = []
	position: Optional[Trade] = None

	# Pre-compute annualized vol over rolling window as proxy for IV
	# We'll use a smoothened series to avoid jumps
	log_ret = np.log(df["Close"]).diff()
	vol_bar = log_ret.rolling(100).std()
	ann_factor = math.sqrt(25 * 252)
	ann_vol_series = (vol_bar * ann_factor).fillna(method="bfill").fillna(0.18).clip(0.05, 0.8)

	equity = []
	cum_pnl = 0.0

	for ts_utc, row in df.iterrows():
		price = float(row["Close"])  # underlying
		ts_ist = to_ist(pd.Timestamp(ts_utc))
		expiry = next_thursday_expiry_ist(ts_ist)
		T_years = max((expiry - ts_ist).total_seconds(), 0) / (365.0 * 24 * 3600)
		iv = float(ann_vol_series.loc[ts_utc]) if ts_utc in ann_vol_series.index else 0.18
		strike = round_to_nearest_strike(price)

		# Option pricing function per bar
		def price_option(direction: str) -> float:
			opt_type = "C" if direction == "CALL" else "P"
			return black_scholes_price(price, strike, T_years, RISK_FREE_RATE, iv, opt_type)

		# Exit logic for open position
		if position is not None:
			current_prem = price_option(position.direction)
			# SL/TP
			if current_prem <= position.entry_price * (1 - sl_pct):
				position.exit_price = current_prem
				position.exit_time = ts_ist
				position.reason_exit = "SL"
				cum_pnl += position.pnl() or 0.0
				trades.append(position)
				position = None
			elif current_prem >= position.entry_price * (1 + tp_pct):
				position.exit_price = current_prem
				position.exit_time = ts_ist
				position.reason_exit = "TP"
				cum_pnl += position.pnl() or 0.0
				trades.append(position)
				position = None
			# Time-based exit: end-of-day ~ 15:25 IST or expiry
			elif ts_ist.time() >= pd.Timestamp("15:25").time() or T_years <= 1e-6:
				position.exit_price = current_prem
				position.exit_time = ts_ist
				position.reason_exit = "EOD/EXP"
				cum_pnl += position.pnl() or 0.0
				trades.append(position)
				position = None
			else:
				# Safety max holding
				if position.entry_time.tz_convert("Asia/Kolkata") + pd.Timedelta(days=max_holding_days) <= ts_ist:
					position.exit_price = current_prem
					position.exit_time = ts_ist
					position.reason_exit = "MAX_HOLD"
					cum_pnl += position.pnl() or 0.0
					trades.append(position)
					position = None

		# Entry logic (only one position at a time)
		if position is None and T_years > 0:
			if bool(row.get("buy_call", False)):
				prem = price_option("CALL")
				position = Trade(
					direction="CALL",
					entry_time=ts_ist,
					exit_time=None,
					strike=strike,
					expiry=expiry,
					qty_lots=lots,
					entry_price=prem,
				)
			elif bool(row.get("buy_put", False)):
				prem = price_option("PUT")
				position = Trade(
					direction="PUT",
					entry_time=ts_ist,
					exit_time=None,
					strike=strike,
					expiry=expiry,
					qty_lots=lots,
					entry_price=prem,
				)

		equity.append({
			"ts": ts_ist,
			"cum_pnl": cum_pnl,
		})

	# If position remains at the end, close at last bar price
	if position is not None:
		last_ts_ist = to_ist(pd.Timestamp(df.index[-1]))
		expiry = position.expiry
		T_years = max((expiry - last_ts_ist).total_seconds(), 0) / (365.0 * 24 * 3600)
		price = float(df["Close"].iloc[-1])
		iv = realized_vol_lookback(df)
		strike = position.strike
		opt_type = "C" if position.direction == "CALL" else "P"
		current_prem = black_scholes_price(price, strike, T_years, RISK_FREE_RATE, iv, opt_type)
		position.exit_price = current_prem
		position.exit_time = last_ts_ist
		position.reason_exit = "END"
		cum_pnl += position.pnl() or 0.0
		trades.append(position)

	eq_df = pd.DataFrame(equity).set_index("ts")
	return trades, eq_df


def fetch_data(interval: str = "15m", period: str = "60d") -> pd.DataFrame:
	df = yf.download(INDEX_TICKER, interval=interval, period=period, progress=False)
	if df.empty:
		raise RuntimeError("No data returned. Check internet or yfinance limits.")
	df = df.dropna()
	return df


def summarize_trades(trades: List[Trade]) -> pd.DataFrame:
	rows = []
	for t in trades:
		rows.append({
			"entry_time": t.entry_time,
			"exit_time": t.exit_time,
			"dir": t.direction,
			"strike": t.strike,
			"expiry": t.expiry,
			"entry": t.entry_price,
			"exit": t.exit_price,
			"pnl": t.pnl(),
			"reason": t.reason_exit,
		})
	return pd.DataFrame(rows)


def performance_stats(trade_df: pd.DataFrame, eq_df: pd.DataFrame) -> dict:
	total_pnl = float(trade_df["pnl"].sum()) if not trade_df.empty else 0.0
	wins = int((trade_df["pnl"] > 0).sum()) if not trade_df.empty else 0
	losses = int((trade_df["pnl"] <= 0).sum()) if not trade_df.empty else 0
	win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0

	# Max drawdown on equity
	curve = eq_df["cum_pnl"].values if not eq_df.empty else np.array([0.0])
	if len(curve) == 0:
		mdd = 0.0
	else:
		peaks = np.maximum.accumulate(curve)
		drawdowns = peaks - curve
		mdd = float(drawdowns.max())

	return {
		"trades": int(len(trade_df)),
		"wins": wins,
		"losses": losses,
		"win_rate_pct": round(win_rate, 2),
		"total_pnl_inr": round(total_pnl, 2),
		"max_drawdown_inr": round(mdd, 2),
		"avg_pnl_per_trade_inr": round(total_pnl / len(trade_df), 2) if len(trade_df) else 0.0,
	}


def run_backtest():
	print("Fetching NIFTY data (15m, last 60d)...")
	df = fetch_data("15m", "60d")
	df = compute_indicators(df)
	# Remove warm-up
	df = df.dropna(subset=["rsi", "macd", "macd_signal", "sma20"]) 

	print("Running simulation...")
	trades, eq = simulate_backtest(df, sl_pct=0.3, tp_pct=0.6, max_holding_days=2, lots=1)
	tdf = summarize_trades(trades)
	stats = performance_stats(tdf, eq)

	print("\nSummary:")
	for k, v in stats.items():
		print(f"- {k}: {v}")

	if not tdf.empty:
		print("\nSample last 5 trades:")
		print(tdf.tail(5).to_string(index=False))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="NIFTY options backtester and NSE option-chain fetcher")
	sub = parser.add_subparsers(dest="cmd")

	# Backtest
	sub.add_parser("backtest", help="Run strategy backtest")

	# Option chain fetch
	p_chain = sub.add_parser("option-chain", help="Fetch NSE option chain with optional expiry filter")
	p_chain.add_argument("symbol", help="Symbol for option chain (e.g., NIFTY, BANKNIFTY, RELIANCE)")
	p_chain.add_argument("--expiry", dest="expiry", help="Expiry in DD-MMM-YYYY format", default=None)
	p_chain.add_argument("--equity", action="store_true", help="Use equities endpoint instead of indices")
	p_chain.add_argument("--csv", dest="csv", help="Path to save CSV")

	# Strategy recommendation
	p_sig = sub.add_parser("signal", help="Analyze option chain and print a trade recommendation")
	p_sig.add_argument("symbol", help="Symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)")
	p_sig.add_argument("--expiry", dest="expiry", help="Expiry DD-MMM-YYYY", default=None)
	p_sig.add_argument("--equity", action="store_true", help="Use equities endpoint")
	p_sig.add_argument("--sl", type=float, default=0.3, help="Stop-loss pct (default 0.3)")
	p_sig.add_argument("--tp", type=float, default=0.6, help="Take-profit pct (default 0.6)")
	p_sig.add_argument("--out", help="Write recommendation JSON to this file")

	args = parser.parse_args()

	def _normalize_expiry(exp: Optional[str], exps: Optional[List[str]] = None) -> Optional[str]:
		if not exp:
			return None
		try:
			dt = dateparser.parse(exp, dayfirst=True, fuzzy=True)
			fmt = dt.strftime("%d-%b-%Y")  # e.g., 21-Aug-2025
		except Exception:
			fmt = exp
		if exps:
			# Exact match first
			if fmt in exps:
				return fmt
			# Try date-equivalence match
			for e in exps:
				try:
					ed = dateparser.parse(e, dayfirst=True, fuzzy=True)
					if ed.date() == dt.date():
						return e
				except Exception:
					continue
		return fmt
	if not args.cmd or args.cmd == "backtest":
		try:
			run_backtest()
		except Exception as e:
			print(f"Error: {e}")
			sys.exit(1)
	elif args.cmd == "option-chain":
		if NSEClient is None:
			print("Option-chain client not available. Please ensure 'requests' is installed.")
			sys.exit(2)
		client = NSEClient.create(args.symbol)
		try:
			# First fetch without expiry to obtain valid expiry list
			payload0 = (
				client.get_option_chain_equities(args.symbol, None)
				if args.equity
				else client.get_option_chain_indices(args.symbol, None)
			)
			exps = list_expiry_dates(payload0) if list_expiry_dates else []
			norm_exp = _normalize_expiry(args.expiry, exps)
			payload = (
				client.get_option_chain_equities(args.symbol, norm_exp)
				if args.equity
				else client.get_option_chain_indices(args.symbol, norm_exp)
			)
		except Exception as e:
			print(f"Failed to fetch option chain: {e}")
			sys.exit(3)

		exps = list_expiry_dates(payload) if list_expiry_dates else []
		if exps:
			print("Available expiries:")
			for x in exps[:12]:  # cap print
				print("-", x)
		if args.expiry:
			disp = _normalize_expiry(args.expiry, exps)
			if disp not in exps:
				print(f"Warning: requested expiry '{args.expiry}' not in available expiries.")

		df = option_chain_to_df(payload, only_expiry=_normalize_expiry(args.expiry, exps)) if option_chain_to_df else pd.DataFrame()
		print("\nPreview:")
		print(df.head(20).to_string(index=False))
		if args.csv:
			df.to_csv(args.csv, index=False)
			print(f"Saved to {args.csv}")
	elif args.cmd == "signal":
		if NSEClient is None or analyze_option_chain is None:
			print("Missing components. Ensure dependencies are installed.")
			sys.exit(2)
		client = NSEClient.create(args.symbol)
		try:
			payload0 = (
				client.get_option_chain_equities(args.symbol, None)
				if args.equity
				else client.get_option_chain_indices(args.symbol, None)
			)
			exps = list_expiry_dates(payload0) if list_expiry_dates else []
			norm_exp = _normalize_expiry(args.expiry, exps)
			payload = (
				client.get_option_chain_equities(args.symbol, norm_exp)
				if args.equity
				else client.get_option_chain_indices(args.symbol, norm_exp)
			)
		except Exception as e:
			print(f"Failed to fetch option chain: {e}")
			sys.exit(3)

		rec = analyze_option_chain(args.symbol, payload, only_expiry=norm_exp, sl_pct=args.sl, tp_pct=args.tp)
		# Print
		print("Recommendation:")
		print(f"- Symbol: {rec.symbol}")
		print(f"- Expiry: {rec.expiry}")
		print(f"- Underlying: {rec.underlying}")
		print(f"- PCR (OI): {rec.pcr_oi:.2f}")
		print(f"- Support (PE OI max): {rec.support_strike}")
		print(f"- Resistance (CE OI max): {rec.resistance_strike}")
		if rec.direction:
			print(f"- Trade: {rec.direction} {rec.strike} | LTP={rec.ltp} | IV={rec.iv}")
			print(f"- Risk: SL={int(rec.stop_loss_pct*100)}% | TP={int(rec.take_profit_pct*100)}%")
		else:
			print("- Trade: No trade")
		print(f"- Reason: {rec.reason}")

		# Optional write to file
		if args.out:
			try:
				with open(args.out, "w", encoding="utf-8") as f:
					json.dump({
						"symbol": rec.symbol,
						"expiry": rec.expiry,
						"underlying": rec.underlying,
						"pcr_oi": rec.pcr_oi,
						"support_strike": rec.support_strike,
						"resistance_strike": rec.resistance_strike,
						"direction": rec.direction,
						"strike": rec.strike,
						"ltp": rec.ltp,
						"iv": rec.iv,
						"stop_loss_pct": rec.stop_loss_pct,
						"take_profit_pct": rec.take_profit_pct,
						"reason": rec.reason,
					}, f, ensure_ascii=False, indent=2)
				print(f"Saved: {args.out}")
			except Exception as e:
				print(f"Failed to write output file: {e}")

