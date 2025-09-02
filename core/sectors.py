# /sp500_screener/core/sectors.py
from __future__ import annotations
import numpy as np
import pandas as pd

def sector_medians(df: pd.DataFrame) -> dict:
    """Medianas por sector para métricas clave (para comparativas y scoring)."""
    metrics = [
        "pe_ttm","ev_ebitda_ttm","p_fcf","p_b","de_ratio","current_ratio",
        "interest_coverage","fcf_margin","cfo_ni_ratio",
    ]
    out = {}
    for sector, sub in df.groupby("Sector"):
        out[sector] = {f"{m}_median": (np.nanmedian(sub[m].values) if m in sub.columns else np.nan) for m in metrics}
    return out
