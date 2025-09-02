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
    # dropna=False para no perder filas con Sector NaN (por si acaso)
    for sector, sub in df.groupby("Sector", dropna=False):
        out[sector] = {}
        for m in metrics:
            if m in sub.columns:
                # to_numeric + median(skipna=True) => sin warnings si todo es NaN
                med = pd.to_numeric(sub[m], errors="coerce").median(skipna=True)
            else:
                med = np.nan
            out[sector][f"{m}_median"] = float(med) if pd.notna(med) else np.nan
    return out
