"""
NSE Option Chain client with expiry-date filtering.

Notes:
- NSE protects its APIs; we first load a landing page to get cookies.
- We use realistic headers and maintain a session.
- Endpoints:
  - Indices: https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY[&date=DD-MMM-YYYY]
  - Equities: https://www.nseindia.com/api/option-chain-equities?symbol=RELIANCE[&date=DD-MMM-YYYY]
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.nseindia.com"


def _default_headers(symbol: str) -> Dict[str, str]:
    # Common headers to mimic a browser
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
    "Accept": "application/json, text/plain, */*",
    # Avoid brotli ('br') so requests can auto-decode; keep gzip/deflate
    "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": f"{BASE}/option-chain?symbol={symbol}",
        "Origin": BASE,
        "Host": "www.nseindia.com",
        "Connection": "keep-alive",
    }


@dataclass
class NSEClient:
    session: requests.Session

    @classmethod
    def create(cls, symbol_for_headers: str = "NIFTY") -> "NSEClient":
        s = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update(_default_headers(symbol_for_headers))
        return cls(session=s)

    def _prime_cookies(self, symbol: str) -> None:
        # Hit landing pages to receive cookies
        for url in [f"{BASE}/", f"{BASE}/option-chain?symbol={symbol}"]:
            try:
                self.session.get(url, timeout=10)
                time.sleep(0.5)
            except requests.RequestException:
                pass

    def _get_json(self, url: str, params: Dict[str, str]) -> Dict:
        r = self.session.get(url, params=params, timeout=20)
        ct = r.headers.get("Content-Type", "")
        ce = r.headers.get("Content-Encoding", "")
        if r.status_code != 200:
            preview = (r.text or r.content[:200]).__repr__() if (r.text or r.content) else "''"
            raise RuntimeError(
                f"HTTP {r.status_code} from NSE. Content-Type={ct}; Content-Encoding={ce}. First bytes: {preview}"
            )
        # Try native JSON first
        try:
            return r.json()
        except ValueError:
            # Attempt manual decode for compressed payloads
            raw = r.content or b""
            if not raw:
                raise RuntimeError(
                    f"Empty body from NSE. Content-Type={ct}; Content-Encoding={ce}."
                )
            # Try brotli if indicated
            if "br" in ce.lower():
                try:
                    import brotli  # type: ignore
                    decoded = brotli.decompress(raw)
                    return json.loads(decoded.decode("utf-8"))
                except Exception as e:
                    raise RuntimeError(
                        "Brotli-compressed JSON but brotli decode failed. Install 'brotli' and retry. "
                        f"Content-Type={ct}; Content-Encoding={ce}; Error={e}"
                    )
            # Try gzip
            if "gzip" in ce.lower():
                try:
                    import gzip
                    decoded = gzip.decompress(raw)
                    return json.loads(decoded.decode("utf-8"))
                except Exception as e:
                    raise RuntimeError(
                        f"Gzip-compressed JSON but decompression failed: {e}"
                    )
            # Last resort: decode as utf-8
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                preview = raw[:200].__repr__()
                raise RuntimeError(
                    "NSE returned a non-JSON or un-decodable response (possibly blocked/rate-limited). "
                    f"Content-Type={ct}; Content-Encoding={ce}. First bytes: {preview}"
                )

    def get_option_chain_indices(self, symbol: str, expiry: Optional[str] = None) -> Dict:
        self._prime_cookies(symbol)
        params: Dict[str, str] = {"symbol": symbol}
        if expiry:
            params["date"] = expiry
        url = f"{BASE}/api/option-chain-indices"
        return self._get_json(url, params)

    def get_option_chain_equities(self, symbol: str, expiry: Optional[str] = None) -> Dict:
        self._prime_cookies(symbol)
        params: Dict[str, str] = {"symbol": symbol}
        if expiry:
            params["date"] = expiry
        url = f"{BASE}/api/option-chain-equities"
        return self._get_json(url, params)


def list_expiry_dates(payload: Dict) -> List[str]:
    return payload.get("records", {}).get("expiryDates", []) or []


def option_chain_to_df(payload: Dict, only_expiry: Optional[str] = None) -> pd.DataFrame:
    """Convert NSE option chain JSON to a tidy DataFrame filtered by expiry if provided."""
    rows = []
    data = payload.get("records", {}).get("data", [])
    for item in data:
        expiry = item.get("expiryDate")
        if only_expiry and expiry != only_expiry:
            continue
        strike = item.get("strikePrice")
        for side in ("CE", "PE"):
            leg = item.get(side)
            if not leg:
                continue
            rows.append({
                "expiry": expiry,
                "strike": strike,
                "type": side,
                "ltp": leg.get("lastPrice"),
                "iv": leg.get("impliedVolatility"),
                "oi": leg.get("openInterest"),
                "change_oi": leg.get("changeinOpenInterest"),
                "volume": leg.get("totalTradedVolume"),
                "bidqty": leg.get("bidQty"),
                "bidprice": leg.get("bidprice"),
                "askprice": leg.get("askPrice"),
                "askqty": leg.get("askQty"),
                "underlying": leg.get("underlying"),
                "identifier": leg.get("identifier"),
            })
    return pd.DataFrame(rows)
