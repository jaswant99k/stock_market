from __future__ import annotations

import argparse
from typing import List, Optional
from dateutil import parser as dateparser
import pandas as pd

from nse_option_chain import NSEClient, list_expiry_dates, option_chain_to_df


def normalize_expiry(exp: Optional[str], exps: Optional[List[str]] = None) -> Optional[str]:
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


def main():
    p = argparse.ArgumentParser(description="Fetch NSE option-chain data filtered by expiry")
    p.add_argument("symbol", help="Symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)")
    p.add_argument("--expiry", help="Expiry date (any readable format, e.g., 21-Aug-2025 or 21 aug 2025)")
    p.add_argument("--equity", action="store_true", help="Use equities endpoint")
    p.add_argument("--csv", help="Save CSV to this path")
    args = p.parse_args()

    client = NSEClient.create(args.symbol)

    # First fetch to get valid expiries
    payload0 = (
        client.get_option_chain_equities(args.symbol, None)
        if args.equity
        else client.get_option_chain_indices(args.symbol, None)
    )
    exps = list_expiry_dates(payload0)
    if not exps:
        print("No expiries returned by NSE.")
        return

    if args.expiry:
        norm = normalize_expiry(args.expiry, exps)
    else:
        norm = exps[0]

    if norm not in exps:
        print("Available expiries:")
        for e in exps:
            print("-", e)
        print(f"Requested expiry '{args.expiry}' not available; using '{norm}'.")

    # Fetch filtered chain
    payload = (
        client.get_option_chain_equities(args.symbol, norm)
        if args.equity
        else client.get_option_chain_indices(args.symbol, norm)
    )
    df = option_chain_to_df(payload, only_expiry=norm)

    print(f"Symbol: {args.symbol} | Expiry: {norm} | Rows: {len(df)}")
    if not df.empty:
        cols = [c for c in ["type", "strike", "oi", "change_oi", "iv", "ltp", "bidprice", "askprice", "volume"] if c in df.columns]
        print(df[cols].head(20).to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
