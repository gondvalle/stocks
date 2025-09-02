"""Agregados por sector."""
from __future__ import annotations

import pandas as pd


def sector_medians(df: pd.DataFrame) -> dict:
    metrics = [
        "pe_ttm", "ev_ebitda_ttm", "p_fcf", "p_b", "de_ratio", "current_ratio",
        "interest_coverage", "fcf_margin", "cfo_ni_ratio",
    ]
    out = {}
    for sector, sub in df.groupby("Sector"):
        out[sector] = {m: sub[m].median() for m in metrics if m in sub.columns}
    return out
