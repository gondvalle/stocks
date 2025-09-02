"""Construcción del universo y reporting."""
from __future__ import annotations

import concurrent.futures as cf
from typing import List

import pandas as pd

from .constituents import load_sp500_constituents
from .fetch import (
    get_info,
    get_price_and_target,
)
from .scoring import evaluate_company
from .sectors import sector_medians
from .utils import safe_div


def build_universe(max_names: int = 500) -> pd.DataFrame:
    consts = load_sp500_constituents(limit=max_names)
    rows: List[dict] = []
    for t in consts["Ticker"]:
        info = get_info(t)
        price, target = get_price_and_target(t)
        row = {
            "Ticker": t,
            "Sector": info.get("sector"),
            "price": price,
            "target": target,
            "target_vs_price": safe_div(target, price),
            "pe_vs_sector": info.get("forwardPE", float("nan")),
            "ev_ebitda_vs_sector": info.get("enterpriseToEbitda", float("nan")),
            "p_fcf_vs_sector": info.get("freeCashflow", float("nan")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    meds = sector_medians(df)
    scores = []
    for _, row in df.iterrows():
        score, checks = evaluate_company(row.to_dict(), meds.get(row.get("Sector"), {}))
        scores.append((score, checks))
    df = df.copy()
    df["score"] = [s for s, _ in scores]
    df["checks"] = [c for _, c in scores]
    return df.sort_values("score", ascending=False)


def export_csv(df: pd.DataFrame, path: str = "sp500_top10_fundamental_screen.csv") -> None:
    df.to_csv(path, index=False)
