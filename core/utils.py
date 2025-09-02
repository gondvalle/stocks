# /sp500_screener/core/utils.py
"""Utilidades comunes."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def _is_nan(x):
    try:
        return math.isnan(x)
    except Exception:
        return x != x

def safe_div(a, b):
    try:
        if b is None or b == 0 or _is_nan(b):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def pct_change(a, b):
    if b is None or b == 0 or _is_nan(b):
        return np.nan
    return (a - b) / abs(b)

def nanmean_or_nan(vals):
    vals = [v for v in vals if v is not None and not _is_nan(v)]
    return float(np.mean(vals)) if vals else np.nan

def df_latest_col(df):
    """Devuelve (columna_serie, etiqueta_col) más reciente de un DF de yfinance."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
    cols = list(df.columns)
    try:
        cols_sorted = sorted(cols, key=lambda c: pd.to_datetime(str(c)), reverse=True)
    except Exception:
        cols_sorted = cols
    col = cols_sorted[0]
    return df[col], col

def series_from_df_row(df, key_candidates):
    """Devuelve la serie (a través de columnas) de la fila cuyo índice case con alguna key."""
    if df is None or df.empty:
        return None
    idx = [str(i).lower() for i in df.index]
    for k in key_candidates:
        k_low = k.lower()
        for i, name in enumerate(idx):
            if k_low == name or k_low in name:
                return pd.to_numeric(df.iloc[i], errors="coerce")
    return None
