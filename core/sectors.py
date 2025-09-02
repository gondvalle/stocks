# /sp500_screener/core/sectors.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _nanmedian_coerce(vals) -> float:
    """Convierte a numérico, descarta NaN, devuelve mediana o NaN si no hay datos."""
    s = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    return float(np.nanmedian(s.values)) if not s.empty else np.nan

def sector_medians(df: pd.DataFrame, exclude_ticker: str | None = None) -> dict:
    """
    Medianas por sector para métricas clave.
    - Si exclude_ticker está presente, se excluye esa fila antes de agrupar (evita 'compararte contigo').
    - Tolera NaNs: calcula con los valores disponibles del sector.
    """
    if exclude_ticker:
        df = df[df["Ticker"] != exclude_ticker].copy()

    metrics = [
        "pe_ttm","ev_ebitda_ttm","p_fcf","p_b","de_ratio","current_ratio",
        "interest_coverage","fcf_margin","cfo_ni_ratio",
    ]
    out = {}
    for sector, sub in df.groupby("Sector"):
        out[sector] = {
            f"{m}_median": _nanmedian_coerce(sub[m].values) if m in sub.columns else np.nan
            for m in metrics
        }
    return out
