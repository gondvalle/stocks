# /sp500_screener/core/constituents.py
from __future__ import annotations
import os, io, requests, pandas as pd
from .constants import SP500_GITHUB_CSV, LOCAL_SP500_CSV

def _get_csv(url: str, timeout: int = 20) -> pd.DataFrame | None:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None

def load_sp500_constituents(local_csv: str = LOCAL_SP500_CSV, limit: int | None = None) -> pd.DataFrame:
    """Devuelve DF con columnas: Ticker, Name, Sector (evitando Wikipedia)."""
    cols_norm = {"symbol":"Ticker","Symbol":"Ticker","ticker":"Ticker",
                 "name":"Name","Name":"Name","sector":"Sector","Sector":"Sector"}
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv).rename(columns=cols_norm)
        if "Ticker" not in df.columns:
            raise ValueError("El CSV local debe tener columna 'Ticker'.")
        return df if limit is None else df.head(limit)

    df = _get_csv(SP500_GITHUB_CSV)
    if df is not None and "Symbol" in df.columns:
        df = df.rename(columns=cols_norm)
        return df if limit is None else df.head(limit)

    fallback = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","JPM","XOM","UNH","HD","LLY","AVGO","BRK-B","V","MA","PG"]
    return pd.DataFrame({"Ticker": fallback if limit is None else fallback[:limit]})
