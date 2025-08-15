from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
import streamlit as st
from dateutil import parser as dateparser

from nse_option_chain import NSEClient, list_expiry_dates, option_chain_to_df
from strategies import analyze_option_chain

st.set_page_config(page_title="NSE Option-Chain Recommender", layout="wide")
st.title("NSE Option-Chain Recommender")

@st.cache_data(ttl=90)
def _fetch_expiries(symbol: str, equity: bool) -> List[str]:
    client = NSEClient.create(symbol)
    payload = (
        client.get_option_chain_equities(symbol, None)
        if equity
        else client.get_option_chain_indices(symbol, None)
    )
    return list_expiry_dates(payload)

@st.cache_data(ttl=30)
def _fetch_chain(symbol: str, equity: bool, expiry: Optional[str]) -> Tuple[Dict, pd.DataFrame]:
    client = NSEClient.create(symbol)
    payload = (
        client.get_option_chain_equities(symbol, expiry)
        if equity
        else client.get_option_chain_indices(symbol, expiry)
    )
    df = option_chain_to_df(payload, only_expiry=expiry)
    return payload, df


def _normalize_exp(exp: Optional[str], exps: Optional[List[str]]) -> Optional[str]:
    if not exp:
        return None
    try:
        dt = dateparser.parse(exp, dayfirst=True, fuzzy=True)
        fmt = dt.strftime("%d-%b-%Y")
    except Exception:
        fmt = exp
    if exps:
        if fmt in exps:
            return fmt
        for e in exps:
            try:
                ed = dateparser.parse(e, dayfirst=True, fuzzy=True)
                if ed.date() == dateparser.parse(fmt, dayfirst=True, fuzzy=True).date():
                    return e
            except Exception:
                continue
    return fmt

with st.sidebar:
    st.header("Controls")
    market_type = st.radio("Market", ["Index", "Equity"], horizontal=True)
    is_equity = market_type == "Equity"
    if is_equity:
        symbol = st.text_input("Equity Symbol", value="RELIANCE").upper().strip()
    else:
        symbol = st.selectbox("Index Symbol", ["NIFTY", "BANKNIFTY"], index=0)

    # Fetch expiries
    expiries: List[str] = []
    err_exp = None
    try:
        expiries = _fetch_expiries(symbol, is_equity)
    except Exception as e:
        err_exp = str(e)

    default_exp_idx = 0 if expiries else None
    expiry_sel = st.selectbox("Expiry", expiries, index=default_exp_idx, key="expiry_sel") if expiries else None

    sl = st.slider("Stop-loss %", min_value=10, max_value=60, value=30, step=5)
    tp = st.slider("Take-profit %", min_value=20, max_value=200, value=60, step=10)

    # Refresh controls
    st.markdown("### Refresh")
    colr1, colr2 = st.columns([1, 1])
    with colr1:
        manual = st.button("Refresh now", type="primary")
    with colr2:
        auto = st.checkbox("Auto refresh", value=False, key="auto_refresh")
        interval = st.selectbox("Interval (sec)", [15, 30, 60, 120], index=1, key="refresh_interval")
    if manual:
        st.cache_data.clear()
        st.experimental_rerun()

if err_exp:
    st.warning(f"Failed to load expiries: {err_exp}")

if not symbol:
    st.stop()

norm_exp = _normalize_exp(expiry_sel, expiries)

payload: Dict
chain_df: pd.DataFrame
try:
    payload, chain_df = _fetch_chain(symbol, is_equity, norm_exp)
except Exception as e:
    st.error(f"Failed to fetch option chain: {e}")
    st.stop()

rec = analyze_option_chain(symbol, payload, only_expiry=norm_exp, sl_pct=sl/100.0, tp_pct=tp/100.0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Underlying", f"{rec.underlying:,.2f}")
col2.metric("PCR (OI)", f"{rec.pcr_oi:.2f}")
col3.metric("Support (PE OI max)", f"{rec.support_strike if rec.support_strike else '-'}")
col4.metric("Resistance (CE OI max)", f"{rec.resistance_strike if rec.resistance_strike else '-'}")

st.subheader("Recommendation")
if rec.direction:
    st.success(
        f"{rec.direction} {rec.strike} | LTP={rec.ltp} | IV={rec.iv} | "
        f"SL={int(rec.stop_loss_pct*100)}% | TP={int(rec.take_profit_pct*100)}%\n\nReason: {rec.reason}"
    )
else:
    st.info(f"No trade | {rec.reason}")

st.subheader("Option Chain (filtered by expiry)")
if chain_df.empty:
    st.write("No rows.")
else:
    # Wider view with sorting
    sort_cols = ["type", "strike", "oi", "change_oi", "iv", "ltp", "bidprice", "askprice", "volume"]
    view_cols = [c for c in sort_cols if c in chain_df.columns]
    st.dataframe(chain_df[view_cols].sort_values(["type", "strike"]).reset_index(drop=True), use_container_width=True)

    # Download button
    csv_buf = io.StringIO()
    chain_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buf.getvalue(),
        file_name=f"{symbol}_{norm_exp}_option_chain.csv".replace("/", "-"),
        mime="text/csv",
    )

# Auto-refresh loop trigger (single-shot per load)
if st.session_state.get("auto_refresh"):
    # Do not block UI excessively; schedule a rerun after the interval
    time.sleep(int(st.session_state.get("refresh_interval", 30)))
    st.experimental_rerun()
