"""Cliente yfinance con caché en disco."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import math
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    class _YF:
        def __getattr__(self, _):
            raise RuntimeError("yfinance no disponible")
    yf = _YF()

from .config import CACHE_DIR

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def disk_cache(ttl_hours: int = 24) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(ticker: str):
            cache_file = CACHE_DIR / f"{func.__name__}_{ticker}.json"
            if cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl_hours * 3600:
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                    if data.get("type") == "dataframe" and pd is not None:
                        return pd.read_json(data["value"], orient="split")
                    else:
                        return data["value"]
                except Exception:
                    pass
            result = func(ticker)
            try:
                if pd is not None and isinstance(result, pd.DataFrame):
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
def get_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


@disk_cache()
def get_income_stmt(ticker: str):
    try:
        return yf.Ticker(ticker).income_stmt
    except Exception:
        return pd.DataFrame() if pd is not None else {}


@disk_cache()
def get_balance_sheet(ticker: str):
    try:
        return yf.Ticker(ticker).balance_sheet
    except Exception:
        return pd.DataFrame() if pd is not None else {}


@disk_cache()
def get_cashflow(ticker: str):
    try:
        return yf.Ticker(ticker).cashflow
    except Exception:
        return pd.DataFrame() if pd is not None else {}


@disk_cache()
def get_shares_history(ticker: str) -> object:
    try:
        return yf.Ticker(ticker).get_shares_full()
    except Exception:
        return None


def get_price_and_target(ticker: str) -> tuple[float | float, float | float]:
    try:
        tk = yf.Ticker(ticker)
        price = tk.fast_info.get("lastPrice") if hasattr(tk, "fast_info") else None
        if price is None:
            hist = tk.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if hasattr(hist, "empty") and not hist.empty else math.nan
        target = tk.info.get("targetMeanPrice", math.nan)
        return price, target
    except Exception:
        return math.nan, math.nan


def get_price_history(ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="5y")
        return hist["Close"]
    except Exception:
        if pd is not None:
            return pd.Series(dtype=float)
        return []
