# /sp500_screener/core/metrics.py
"""Cálculo de métricas financieras (replica tu script monolítico)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime

from .utils import safe_div, pct_change, nanmean_or_nan, df_latest_col, series_from_df_row
from .config import RISK_FREE
import yfinance as yf

def compute_history_pe_yf(ticker, info, income, shares, years=10):
    """PE históricos 5-10y con EPS anual ≈ Net Income / Shares y precio fin de año."""
    try:
        eps_series = None

        ni = series_from_df_row(income, ["Net Income", "NetIncome"])
        so_hist = None
        if isinstance(shares, pd.DataFrame) and not shares.empty:
            try:
                sh = shares.reset_index()
                if "Date" in sh.columns and "Shares Outstanding" in sh.columns:
                    sh["Date"] = pd.to_datetime(sh["Date"]).dt.year
                    sh = sh.drop_duplicates("Date").set_index("Date")["Shares Outstanding"]
                    so_hist = sh
            except Exception:
                so_hist = None
        if (so_hist is None or so_hist.empty) and info:
            so_hist = pd.Series({datetime.now().year: info.get("sharesOutstanding", np.nan)})

        if ni is not None and so_hist is not None and not so_hist.empty:
            years_idx = []
            for c in income.columns:
                try:
                    years_idx.append(pd.to_datetime(str(c)).year)
                except Exception:
                    years_idx.append(None)
            eps_vals, years_valid = [], []
            for j, col in enumerate(income.columns):
                y = years_idx[j]
                if y is None:
                    continue
                ni_y = ni[col]
                so_y = so_hist.loc[y] if y in so_hist.index else so_hist.iloc[-1]
                if pd.notna(ni_y) and pd.notna(so_y) and so_y > 0:
                    eps_vals.append(float(ni_y) / float(so_y))
                    years_valid.append(y)
            if eps_vals:
                eps_series = pd.Series(eps_vals, index=years_valid).sort_index()

        if eps_series is None or eps_series.empty:
            return np.nan, np.nan

        start_year = max(eps_series.index.min(), datetime.now().year - years - 1)
        price_hist = yf.Ticker(ticker).history(period="max")["Close"]
        price_yr = price_hist.resample("Y").last()
        price_yr.index = price_yr.index.year
        pe_list = []
        for y, eps in eps_series.items():
            if y < start_year:
                continue
            px = price_yr[price_yr.index == y]
            if not px.empty and pd.notna(eps) and eps > 0:
                pe_list.append(float(px.iloc[0]) / float(eps))
        if not pe_list:
            return np.nan, np.nan
        pe_series = pd.Series(pe_list)
        pe10 = float(pe_series.tail(min(10, len(pe_series))).mean())
        pe5 = float(pe_series.tail(min(5, len(pe_series))).mean())
        return pe10, pe5
    except Exception:
        return np.nan, np.nan

def compute_margins_and_trends(income, years_window=5):
    out = {}
    if income is None or income.empty:
        return out
    rev = series_from_df_row(income, ["Total Revenue","Revenue"])
    gp  = series_from_df_row(income, ["Gross Profit"])
    ebit = series_from_df_row(income, ["EBIT","Operating Income"])
    ni  = series_from_df_row(income, ["Net Income","NetIncome"])

    def last_and_trend(num, den):
        last = np.nan; tr = np.nan
        if num is not None and den is not None:
            m = pd.to_numeric(num, errors="coerce")/pd.to_numeric(den, errors="coerce")
            m = m.dropna()
            if not m.empty:
                last = float(m.iloc[-1])
                base = m.iloc[max(0, len(m)-years_window-1)]
                tr = pct_change(m.iloc[-1], base)
        return last, tr

    g_last, g_tr = last_and_trend(gp, rev)
    o_last, o_tr = last_and_trend(ebit, rev)
    n_last, n_tr = last_and_trend(ni, rev)

    out.update({
        "grossProfitMargin_last": g_last, "grossProfitMargin_trend": g_tr,
        "operatingProfitMargin_last": o_last, "operatingProfitMargin_trend": o_tr,
        "netProfitMargin_last": n_last, "netProfitMargin_trend": n_tr,
    })
    return out

def compute_growth(income):
    out = {}
    if income is None or income.empty:
        return out
    rev = series_from_df_row(income, ["Total Revenue","Revenue"])
    eps = series_from_df_row(income, ["Diluted EPS","Basic EPS"])
    def cagr(series, yrs=5):
        if series is None or series.dropna().shape[0] < 2:
            return np.nan
        s = series.dropna().astype(float)
        first = s.iloc[max(0, len(s)-yrs-1)]
        last  = s.iloc[-1]
        years = min(yrs, len(s)-1)
        if first > 0 and last > 0 and years > 0:
            return (last/first)**(1/years) - 1
        return np.nan
    out["rev_cagr"] = float(cagr(rev))
    out["eps_cagr"] = float(cagr(eps)) if eps is not None else np.nan
    return out

def compute_operating_leverage(income):
    out = {"op_leverage": np.nan}
    if income is None or income.empty:
        return out
    rev = series_from_df_row(income, ["Total Revenue","Revenue"])
    ebitda = series_from_df_row(income, ["EBITDA","Ebitda"])
    if rev is None or ebitda is None:
        return out
    def growth_last(series, yrs=4):
        s = series.dropna().astype(float)
        if s.shape[0] < 2:
            return np.nan
        first = s.iloc[max(0, len(s)-yrs-1)]
        last = s.iloc[-1]
        return pct_change(last, first)
    e_g = growth_last(ebitda); r_g = growth_last(rev)
    out["op_leverage"] = float(e_g - r_g) if pd.notna(e_g) and pd.notna(r_g) else np.nan
    return out

def compute_cash_quality_and_fcf(income, cashflow):
    out = {"cfo_ni_ratio": np.nan, "fcf_margin": np.nan}
    if cashflow is None or cashflow.empty or income is None or income.empty:
        return out
    cf_col, _ = df_latest_col(cashflow)
    inc_col, _ = df_latest_col(income)
    if cf_col is None or inc_col is None:
        return out
    cfo = float(cf_col.get("Operating Cash Flow", np.nan) or cf_col.get("Total Cash From Operating Activities", np.nan) or np.nan)
    capex = float(cf_col.get("Capital Expenditure", np.nan) or cf_col.get("Capital Expenditures", np.nan) or np.nan)
    ni = float(inc_col.get("Net Income", np.nan) or inc_col.get("NetIncome", np.nan) or np.nan)
    rev = float(inc_col.get("Total Revenue", np.nan) or inc_col.get("Revenue", np.nan) or np.nan)
    if pd.notna(cfo) and pd.notna(ni) and ni != 0:
        out["cfo_ni_ratio"] = cfo/ni
    if pd.notna(cfo) and pd.notna(capex):
        fcf = cfo + capex
        if pd.notna(rev) and rev != 0:
            out["fcf_margin"] = fcf/rev
    return out

def compute_balance_ratios(income, balance, info):
    out = {"de_ratio": np.nan, "interest_coverage": np.nan, "current_ratio": np.nan,
           "p_b": np.nan, "p_fcf": np.nan, "ev_ebitda_ttm": np.nan, "roe": np.nan, "roa": np.nan}
    bal_col, _ = df_latest_col(balance) if balance is not None and not balance.empty else (None, None)
    inc_col, _ = df_latest_col(income) if income is not None and not income.empty else (None, None)

    if bal_col is not None:
        td = float(bal_col.get("Total Debt", np.nan) or
                   ((bal_col.get("Short Long Term Debt", np.nan) or 0) + (bal_col.get("Long Term Debt", np.nan) or 0)))
        te = float(bal_col.get("Total Stockholder Equity", np.nan) or bal_col.get("Total Equity", np.nan) or np.nan)
        ca = float(bal_col.get("Total Current Assets", np.nan) or np.nan)
        cl = float(bal_col.get("Total Current Liabilities", np.nan) or np.nan)
        ta = float(bal_col.get("Total Assets", np.nan) or np.nan)
        out["de_ratio"] = safe_div(td, te)
        out["current_ratio"] = safe_div(ca, cl)
        if inc_col is not None:
            out["roa"] = safe_div(inc_col.get("Net Income", np.nan), ta)

    if inc_col is not None:
        ebit = float(inc_col.get("EBIT", np.nan) or inc_col.get("Operating Income", np.nan) or np.nan)
        interest_exp = inc_col.get("Interest Expense", np.nan)
        if pd.notna(interest_exp): interest_exp = abs(float(interest_exp))
        out["interest_coverage"] = safe_div(ebit, interest_exp)

    if inc_col is not None and bal_col is not None:
        ni = float(inc_col.get("Net Income", np.nan) or np.nan)
        te = float(bal_col.get("Total Stockholder Equity", np.nan) or np.nan)
        out["roe"] = safe_div(ni, te)

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    mcap = info.get("marketCap")
    total_debt = info.get("totalDebt")
    cash = info.get("totalCash")
    ebitda_info = info.get("ebitda")

    if pd.notna(mcap):
        ev = mcap + (total_debt or 0) - (cash or 0)
        if ebitda_info and ebitda_info != 0:
            out["ev_ebitda_ttm"] = ev / ebitda_info

    if pd.notna(mcap) and bal_col is not None:
        te = float(bal_col.get("Total Stockholder Equity", np.nan) or np.nan)
        out["p_b"] = safe_div(mcap, te)

    return out

def compute_roic_wacc_proxy(income, balance, info):
    out = {"roic": np.nan, "wacc_proxy": np.nan}
    if income is None or income.empty or balance is None or balance.empty:
        return out
    inc_col, _ = df_latest_col(income)
    bal_col, _ = df_latest_col(balance)
    if inc_col is None or bal_col is None:
        return out
    ebit = float(inc_col.get("EBIT", np.nan) or inc_col.get("Operating Income", np.nan) or np.nan)
    tax_exp = float(inc_col.get("Tax Provision", np.nan) or inc_col.get("Income Tax Expense", np.nan) or np.nan)
    pretax = float(inc_col.get("Income Before Tax", np.nan) or inc_col.get("Pretax Income", np.nan) or np.nan)
    if pd.notna(tax_exp) and pd.notna(pretax) and pretax>0:
        tax_rate = min(0.5, max(0.0, tax_exp/pretax))
    else:
        tax_rate = 0.21
    cash = float(bal_col.get("Cash And Cash Equivalents", np.nan) or bal_col.get("Cash", np.nan) or (info.get("totalCash") or np.nan))
    debt = float(bal_col.get("Total Debt", np.nan) or ((bal_col.get("Short Long Term Debt", 0) or 0)+(bal_col.get("Long Term Debt",0) or 0)))
    equity = float(bal_col.get("Total Stockholder Equity", np.nan) or np.nan)
    invested_cap = (debt if pd.notna(debt) else 0) + (equity if pd.notna(equity) else 0) - (cash if pd.notna(cash) else 0)
    if pd.notna(ebit) and pd.notna(invested_cap) and invested_cap>0:
        nopat = ebit * (1 - tax_rate)
        out["roic"] = nopat / invested_cap

    beta = info.get("beta") or 1.0
    de_ratio = np.nan
    if pd.notna(debt) and pd.notna(equity) and equity>0:
        de_ratio = debt/equity
    try:
        Re = RISK_FREE + (beta)*0.05
        Rd = RISK_FREE + 0.02
        D_over_E = max(0.0, de_ratio) if pd.notna(de_ratio) else 0.5
        wD = D_over_E/(1.0 + D_over_E)
        wE = 1.0 - wD
        out["wacc_proxy"] = wE*Re + wD*Rd*(1-0.21)
    except Exception:
        pass
    return out

def compute_p_fcf(mcap: float | None, income, cashflow):
    if mcap is None or pd.isna(mcap) or cashflow is None or cashflow.empty or income is None or income.empty:
        return np.nan
    cf_col, _ = df_latest_col(cashflow)
    if cf_col is None:
        return np.nan
    cfo = float(cf_col.get("Operating Cash Flow", np.nan) or cf_col.get("Total Cash From Operating Activities", np.nan) or np.nan)
    capex = float(cf_col.get("Capital Expenditure", np.nan) or cf_col.get("Capital Expenditures", np.nan) or np.nan)
    if pd.notna(cfo) and pd.notna(capex):
        fcf = cfo + capex
        if fcf != 0:
            return mcap / fcf
    return np.nan

def compute_buybacks(shares):
    if shares is None or shares.empty:
        return np.nan
    try:
        s = shares.reset_index()
        if "Date" in s.columns and "Shares Outstanding" in s.columns:
            s["Date"] = pd.to_datetime(s["Date"]).dt.year
            s = s.drop_duplicates("Date").set_index("Date")["Shares Outstanding"].sort_index()
            if len(s) >= 2:
                first = s.iloc[max(0, len(s)-5)]
                last = s.iloc[-1]
                if first>0:
                    return (last-first)/first
    except Exception:
        return np.nan
    return np.nan
