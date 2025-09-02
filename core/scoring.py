"""Evaluación y scoring de compañías."""
from __future__ import annotations

from typing import Dict, Tuple
import math

THRESHOLDS = {
    "target_vs_price": 1.0,
    "pe_vs_sector": 1.0,
    "ev_ebitda_vs_sector": 1.0,
    "p_fcf_vs_sector": 1.0,
}


def evaluate_company(row: dict, sector_stats: dict) -> Tuple[float, dict]:
    checks = {}
    checks["target"] = row.get("target_vs_price", math.nan) > THRESHOLDS["target_vs_price"]
    checks["pe"] = row.get("pe_vs_sector", math.nan) < THRESHOLDS["pe_vs_sector"]
    checks["ev_ebitda"] = row.get("ev_ebitda_vs_sector", math.nan) < THRESHOLDS["ev_ebitda_vs_sector"]
    checks["p_fcf"] = row.get("p_fcf_vs_sector", math.nan) < THRESHOLDS["p_fcf_vs_sector"]

    valuation_passes = sum([checks["pe"], checks["ev_ebitda"], checks["p_fcf"]])
    hard_gate = checks["target"] and valuation_passes >= 2

    score = float(sum(checks.values())) if hard_gate else 0.0
    return score, checks
