# /sp500_screener/core/fetch.py
"""Cliente yfinance con caché en disco y helpers robustos."""
from __future__ import annotations
import json, time, math
from pathlib import Path
from typing import Callable, Tuple, Any

import pandas as pd
try:
    import yfinance as yf
except Exception:
    class _YF:
        def __getattr__(self, _):
            raise RuntimeError("yfinance no disponible")
    yf = _YF()

from .config import CACHE_DIR

def disk_cache(ttl_hours: int = 24) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            key = func.__name__ + "_" + "_".join([str(a) for a in args])
            cache_file = CACHE_DIR / f"{key}.json"
            if cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl_hours * 3600:
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                    if data.get("type") == "dataframe":
                        return pd.read_json(data["value"], orient="split")
                    return data["value"]
                except Exception:
                    pass
            result = func(*args, **kwargs)
            try:
                if isinstance(result, pd.DataFrame):
                    payload = {"type": "dataframe", "value": result.to_json(orient="split")}
                else:
                    payload = {"type": "object", "value": result}
                with open(cache_file, "w") as f:
                    json.dump(payload, f)
            except Exception:
                pass
            return result
        return wrapper
    return decorator

@disk_cache()
def yf_safe_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

@disk_cache()
def yf_price_and_target(ticker: str) -> Tuple[float, float]:
    try:
        tk = yf.Ticker(ticker)
        price = None
        if hasattr(tk, "fast_info"):
            price = tk.fast_info.get("lastPrice")
        if price is None:
            hist = tk.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else math.nan
        info = {}
        try:
            info = tk.info
        except Exception:
            pass
        target = info.get("targetMeanPrice", math.nan)
        return price, target
    except Exception:
        return math.nan, math.nan

@disk_cache()
def yf_financials(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    out = {"income": None, "balance": None, "cashflow": None, "shares": None, "info": {}}
    try:
        out["income"] = tk.income_stmt
        out["balance"] = tk.balance_sheet
        out["cashflow"] = tk.cashflow
    except Exception:
        pass
    try:
        out["shares"] = tk.get_shares_full()
    except Exception:
        out["shares"] = None
    try:
        out["info"] = tk.info
    except Exception:
        out["info"] = {}
    return out

def get_price_history_close(ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="max")
        return hist["Close"] if "Close" in hist else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)
