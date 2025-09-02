"""Cálculo de métricas financieras (simplificado)."""
from __future__ import annotations

import math
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from .utils import safe_div


def _get(obj, key):
    if obj is None:
        raise KeyError
    if pd and isinstance(obj, pd.DataFrame):
        return obj.loc[key].iloc[0]
    elif isinstance(obj, dict):
        val = obj.get(key)
        if isinstance(val, (list, tuple)):
            return val[0]
        return val
    else:
        raise KeyError


def compute_history_pe_yf(ticker, info, income, shares, years=10):
    try:
        eps = safe_div(_get(income, "Net Income"), _get(shares, "Basic Shares Outstanding"))
        price_hist = info.get("currentPrice", math.nan)
        pe = safe_div(price_hist, eps)
        return pe, pe
    except Exception:
        return math.nan, math.nan


def compute_margins_and_trends(income) -> dict:
    return {"gross_margin": math.nan, "operating_margin": math.nan, "net_margin": math.nan}


def compute_growth(income) -> dict:
    return {"revenue_cagr": math.nan, "eps_cagr": math.nan}


def compute_operating_leverage(income) -> dict:
    return {"operating_leverage": math.nan}


def compute_cash_quality_and_fcf(income, cashflow) -> dict:
    try:
        cfo = _get(cashflow, "Operating Cash Flow")
        ni = _get(income, "Net Income")
        capex = _get(cashflow, "Capital Expenditure")
        fcf = cfo + capex
        rev = _get(income, "Total Revenue")
        return {"cfo_ni_ratio": safe_div(cfo, ni), "fcf_margin": safe_div(fcf, rev)}
    except Exception:
        return {"cfo_ni_ratio": math.nan, "fcf_margin": math.nan}


def compute_balance_ratios(income, balance, info) -> dict:
    return {
        "de_ratio": math.nan,
        "interest_coverage": math.nan,
        "current_ratio": math.nan,
        "p_b": info.get("priceToBook", math.nan),
        "p_fcf": math.nan,
        "ev_ebitda_ttm": info.get("enterpriseToEbitda", math.nan),
        "roe": info.get("returnOnEquity", math.nan),
        "roa": info.get("returnOnAssets", math.nan),
    }


def compute_roic_wacc_proxy(income, balance, info) -> dict:
    return {"roic": info.get("returnOnEquity", math.nan), "wacc_proxy": 0.08}


def compute_p_fcf(market_cap: float | None, income, cashflow) -> float | math.nan:
    try:
        cfo = _get(cashflow, "Operating Cash Flow")
        capex = _get(cashflow, "Capital Expenditure")
        fcf = cfo + capex
        return safe_div(market_cap, fcf)
    except Exception:
        return math.nan


def compute_buybacks(shares) -> float | math.nan:
    try:
        if pd and isinstance(shares, pd.DataFrame):
            series = shares.loc["Basic Shares Outstanding"]
            return safe_div(series.iloc[-1] - series.iloc[0], series.iloc[0])
    except Exception:
        pass
    return math.nan
