"""Utilidades comunes sin dependencias externas."""
from __future__ import annotations

import math


def _is_nan(x):
    try:
        return math.isnan(x)
    except Exception:
        return x != x


def safe_div(a, b):
    try:
        if b is None or b == 0 or _is_nan(b):
            return math.nan
        return a / b
    except Exception:
        return math.nan


def pct_change(a, b):
    if b is None or b == 0 or _is_nan(b):
        return math.nan
    return (a - b) / abs(b)


def nanmean_or_nan(vals):
    vals = [v for v in vals if not _is_nan(v)]
    return sum(vals) / len(vals) if vals else math.nan
