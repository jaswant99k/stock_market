from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class StrategyRecommendation:
    symbol: str
    expiry: str
    underlying: float
    pcr_oi: float
    support_strike: Optional[int]
    resistance_strike: Optional[int]
    direction: Optional[str]  # 'BUY_CALL' | 'BUY_PUT' | None
    strike: Optional[int]
    ltp: Optional[float]
    iv: Optional[float]
    stop_loss_pct: float
    take_profit_pct: float
    reason: str


def compute_pcr(df: pd.DataFrame) -> float:
    ce_oi = df.loc[df.type == "CE", "oi"].fillna(0).sum()
    pe_oi = df.loc[df.type == "PE", "oi"].fillna(0).sum()
    if ce_oi <= 0:
        return 0.0
    return float(pe_oi / ce_oi)


def find_atm_strike(df: pd.DataFrame, underlying: float) -> int:
    strikes = df["strike"].dropna().unique()
    strikes = sorted(int(s) for s in strikes)
    nearest = min(strikes, key=lambda k: abs(k - underlying))
    return int(nearest)


def support_resistance_by_oi(df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    # Support = PE strike with max OI; Resistance = CE strike with max OI
    pe = df[df.type == "PE"]["oi"].fillna(0)
    ce = df[df.type == "CE"]["oi"].fillna(0)
    if pe.empty or ce.empty:
        return None, None
    pe_idx = df[df.type == "PE"].set_index("strike")["oi"].fillna(0)
    ce_idx = df[df.type == "CE"].set_index("strike")["oi"].fillna(0)
    support = int(pe_idx.idxmax()) if not pe_idx.empty else None
    resistance = int(ce_idx.idxmax()) if not ce_idx.empty else None
    return support, resistance


def _row(df: pd.DataFrame, strike: int, opt_type: str) -> pd.Series:
    r = df[(df["strike"] == strike) & (df["type"] == opt_type)]
    return r.iloc[0] if not r.empty else pd.Series(dtype=float)


def analyze_option_chain(symbol: str, payload: Dict, only_expiry: Optional[str] = None,
                         sl_pct: float = 0.3, tp_pct: float = 0.6) -> StrategyRecommendation:
    from nse_option_chain import option_chain_to_df, list_expiry_dates

    exps = list_expiry_dates(payload)
    expiry = only_expiry if (only_expiry and only_expiry in exps) else (exps[0] if exps else "")

    df = option_chain_to_df(payload, only_expiry=expiry)
    if df.empty:
        return StrategyRecommendation(
            symbol=symbol, expiry=expiry, underlying=float(payload.get("records", {}).get("underlyingValue", 0.0)),
            pcr_oi=0.0, support_strike=None, resistance_strike=None,
            direction=None, strike=None, ltp=None, iv=None,
            stop_loss_pct=sl_pct, take_profit_pct=tp_pct,
            reason="No data available"
        )

    underlying = float(payload.get("records", {}).get("underlyingValue", 0.0))
    pcr = compute_pcr(df)
    support, resistance = support_resistance_by_oi(df)
    atm = find_atm_strike(df, underlying)

    ce_atm = _row(df, atm, "CE")
    pe_atm = _row(df, atm, "PE")

    ce_chg_oi = float(ce_atm.get("change_oi", 0)) if not ce_atm.empty else 0.0
    pe_chg_oi = float(pe_atm.get("change_oi", 0)) if not pe_atm.empty else 0.0

    ce_iv = float(ce_atm.get("iv", 0)) if not ce_atm.empty else 0.0
    pe_iv = float(pe_atm.get("iv", 0)) if not pe_atm.empty else 0.0

    ce_ltp = float(ce_atm.get("ltp", 0)) if not ce_atm.empty else 0.0
    pe_ltp = float(pe_atm.get("ltp", 0)) if not pe_atm.empty else 0.0

    # Heuristic rules
    # Bias by PCR
    bullish_bias = pcr >= 1.1
    bearish_bias = pcr <= 0.9

    # Reinforce with OI change at ATM: rising PE OI (support) favors bullish; rising CE OI (resistance) favors bearish
    if bullish_bias and pe_chg_oi > 0 and ce_iv <= 30.0:
        direction = "BUY_CALL"
        strike = atm
        ltp = ce_ltp
        iv = ce_iv
        reason = f"PCR={pcr:.2f} (bullish), PE OI↑ at ATM, IV OK (<=30)"
    elif bearish_bias and ce_chg_oi > 0 and pe_iv <= 30.0:
        direction = "BUY_PUT"
        strike = atm
        ltp = pe_ltp
        iv = pe_iv
        reason = f"PCR={pcr:.2f} (bearish), CE OI↑ at ATM, IV OK (<=30)"
    else:
        direction = None
        strike = None
        ltp = None
        iv = None
        reason = (
            f"No clear edge | PCR={pcr:.2f}, ATM ΔOI: CE={ce_chg_oi:.0f}, PE={pe_chg_oi:.0f}, "
            f"IVs: CE={ce_iv:.1f}, PE={pe_iv:.1f}"
        )

    return StrategyRecommendation(
        symbol=symbol,
        expiry=expiry,
        underlying=underlying,
        pcr_oi=pcr,
        support_strike=support,
        resistance_strike=resistance,
        direction=direction,
        strike=strike,
        ltp=ltp,
        iv=iv,
        stop_loss_pct=sl_pct,
        take_profit_pct=tp_pct,
        reason=reason,
    )
