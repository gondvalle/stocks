# /sp500_screener/core/reporting.py
"""Crea el universo, calcula métricas, scoring y exporta."""
from __future__ import annotations
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .constituents import load_sp500_constituents
from .fetch import yf_safe_info, yf_price_and_target, yf_financials
from .metrics import (
    compute_history_pe_yf, compute_margins_and_trends, compute_growth, compute_operating_leverage,
    compute_cash_quality_and_fcf, compute_balance_ratios, compute_roic_wacc_proxy, compute_p_fcf, compute_buybacks
)
from .sectors import sector_medians
from .scoring import evaluate_company

def _build_row(ticker: str, fallback_sector: str | None, fallback_name: str | None) -> dict:
    info = yf_safe_info(ticker) or {}
    price, target = yf_price_and_target(ticker)
    fin = yf_financials(ticker)
    income, balance, cashflow, shares = fin["income"], fin["balance"], fin["cashflow"], fin["shares"]

    sector = info.get("sector") or fallback_sector or "Unknown"
    name = info.get("longName") or info.get("shortName") or (fallback_name or ticker)
    pe_ttm = info.get("trailingPE")
    forward_pe = info.get("forwardPE")

    pe10, pe5 = compute_history_pe_yf(ticker, info, income, shares, years=10)
    margins = compute_margins_and_trends(income)
    growth = compute_growth(income)
    op_lev = compute_operating_leverage(income)
    cashq = compute_cash_quality_and_fcf(income, cashflow)
    balrat = compute_balance_ratios(income, balance, info)
    p_fcf = compute_p_fcf(info.get("marketCap"), income, cashflow)
    roic_wacc = compute_roic_wacc_proxy(income, balance, info)
    peg = np.nan
    if forward_pe is not None and growth.get("eps_cagr") is not None and growth["eps_cagr"] > 0:
        g = growth["eps_cagr"]
        peg = forward_pe / (g*100 if g < 1 else g)
    sh_trend = compute_buybacks(shares)

    row = {
        "Ticker": ticker, "Name": name, "Sector": sector,
        "price": float(price) if price == price else np.nan,
        "target_mean": float(target) if target == target else np.nan,
        "pe_ttm": float(pe_ttm) if pe_ttm == pe_ttm else np.nan,
        "forward_pe": float(forward_pe) if forward_pe == forward_pe else np.nan,
        "pe_10y": pe10, "pe_5y": pe5,
        "ev_ebitda_ttm": balrat.get("ev_ebitda_ttm"),
        "p_fcf": p_fcf,
        "p_b": balrat.get("p_b"),
        "de_ratio": balrat.get("de_ratio"),
        "interest_coverage": balrat.get("interest_coverage"),
        "current_ratio": balrat.get("current_ratio"),
        "cfo_ni_ratio": cashq.get("cfo_ni_ratio"),
        "fcf_margin": cashq.get("fcf_margin"),
        "roic": roic_wacc.get("roic"),
        "roe": balrat.get("roe"),
        "roa": balrat.get("roa"),
        "op_leverage": op_lev.get("op_leverage"),
        "rev_cagr": growth.get("rev_cagr"),
        "eps_cagr": growth.get("eps_cagr"),
        "shares_out_trend": sh_trend,
        "peg": peg,
        "beta": info.get("beta"),
        "wacc_proxy": roic_wacc.get("wacc_proxy"),
    }
    row.update(margins)
    return row

def build_universe(max_names: int = 500, max_workers: int = 8) -> pd.DataFrame:
    sp = load_sp500_constituents(limit=max_names)
    if "Ticker" not in sp.columns:
        sp["Ticker"] = sp.iloc[:, 0]
    tickers = sp["Ticker"].astype(str).str.replace(".", "-", regex=False).tolist()

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for _, r in sp.iterrows():
            t = str(r["Ticker"]).replace(".", "-")
            futs[ex.submit(_build_row, t, r.get("Sector") if "Sector" in sp.columns else None,
                           r.get("Name") if "Name" in sp.columns else None)] = t
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                pass
    return pd.DataFrame(rows)

def score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    meds = sector_medians(df)
    scores, checks = [], []
    for _, row in df.iterrows():
        s, ch = evaluate_company(row.to_dict(), meds)
        scores.append(s); checks.append(ch)
    out = df.copy()
    out["score"] = scores
    out["checks"] = checks
    return out.sort_values("score", ascending=False)

def export_csv(df: pd.DataFrame, path: str = "sp500_top10_fundamental_screen.csv") -> None:
    cols = [c for c in df.columns if c != "checks"]
    df[cols].to_csv(path, index=False)
