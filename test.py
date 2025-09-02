# -*- coding: utf-8 -*-
"""
Screener fundamental S&P 500 (Top-10) SOLO con fuentes gratuitas (Yahoo/yfinance).
Incluye: precio vs target, forwardPE vs PE histórico 5-10y, PE y EV/EBITDA vs sector,
cashflow vs deuda (CFO/NI, FCF margin, P/FCF), ROIC (aprox) vs WACC proxy, ROE/ROA,
márgenes y tendencias, apalancamiento operativo, D/E, cobertura intereses, current ratio,
buybacks (tendencia de shares), PEG, crecimiento ingresos/EPS, P/B y comparativas.
Imprime informe con “Debería… | Está en… | ✅/⚠️/❌” y exporta CSV.

pip install yfinance pandas numpy python-dateutil requests
"""

import os, io, sys, time, json, warnings, traceback, math
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
# --- imports al inicio del archivo ---
from pathlib import Path
import shutil


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)
warnings.filterwarnings("ignore", category=FutureWarning)

RISK_FREE = float(os.getenv("RISK_FREE", "4.0"))/100.0  # 4% por defecto

# ------------------------ Utilidades ------------------------
def _get_csv(url, timeout=20):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None

def safe_div(a, b):
    try:
        if b is None or b == 0 or pd.isna(b):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def pct_change(a, b):
    if b is None or b == 0 or pd.isna(b):
        return np.nan
    return (a - b) / abs(b)

def nanmean_or_nan(vals):
    vals = [v for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan

# ------------------------ Constituyentes S&P 500 ------------------------
def load_sp500_constituents(local_csv="sp500_constituents.csv", limit=None):
    """1) CSV local; 2) CSV GitHub (datasets/s-and-p-500-companies). Evita Wikipedia."""
    cols_norm = {"symbol":"Ticker", "Symbol":"Ticker", "ticker":"Ticker",
                 "name":"Name", "Name":"Name", "Sector":"Sector", "sector":"Sector"}
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv).rename(columns=cols_norm)
        if "Ticker" not in df.columns:
            raise ValueError("El CSV local debe tener columna 'Ticker' o 'Symbol'.")
        return df if limit is None else df.head(limit)
    gh_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = _get_csv(gh_url)
    if df is not None and "Symbol" in df.columns:
        return df.rename(columns=cols_norm) if limit is None else df.rename(columns=cols_norm).head(limit)
    # fallback mínimo
    fallback = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","JPM","XOM","UNH","HD","LLY","AVGO","BRK.B","V","MA","PG"]
    return pd.DataFrame({"Ticker":fallback}) if limit is None else pd.DataFrame({"Ticker":fallback[:limit]})

# ------------------------ Yahoo helpers ------------------------
def yf_safe_info(t):
    try:
        return yf.Ticker(t).info
    except Exception:
        return {}

def yf_price_and_target(t):
    try:
        tk = yf.Ticker(t)
        price = None
        if hasattr(tk, "fast_info") and "lastPrice" in tk.fast_info:
            price = tk.fast_info["lastPrice"]
        if price is None:
            hist = tk.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else np.nan
        target = None
        try:
            inf = tk.info
            target = inf.get("targetMeanPrice")
        except Exception:
            pass
        return price, target
    except Exception:
        return np.nan, np.nan

def yf_financials(t):
    """Devuelve dict con dfs: income, balance, cashflow (ANUAL), shares_history (anual), info."""
    out = {"income":None, "balance":None, "cashflow":None, "shares":None, "info":{}}
    tk = yf.Ticker(t)
    try:
        inc = tk.income_stmt if hasattr(tk, "income_stmt") else tk.financials
        # yfinance 0.2+ usa .income_stmt / .balance_sheet / .cashflow con DF ya 'nuevos'
        out["income"] = tk.income_stmt
        out["balance"] = tk.balance_sheet
        out["cashflow"] = tk.cashflow
    except Exception:
        pass
    try:
        sh = tk.get_shares_full()  # histórico anual/quarterly MultiIndex
        out["shares"] = sh
    except Exception:
        out["shares"] = None
    try:
        out["info"] = tk.info
    except Exception:
        out["info"] = {}
    return out

def df_latest_col(df):
    """Devuelve la columna más reciente de un DF y su fecha (si existe). En yfinance columnas son fechas."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
    cols = list(df.columns)
    # columnas suelen estar ordenadas de más reciente a más antigua, pero lo forzamos por fecha si posible
    try:
        cols_sorted = sorted(cols, key=lambda c: pd.to_datetime(str(c)), reverse=True)
    except Exception:
        cols_sorted = cols
    col = cols_sorted[0]
    return df[col], col

def series_from_df_row(df, key_candidates):
    """Devuelve la serie temporal (por columnas) de una fila cuyo índice contenga alguna de las keys (case-insensitive)."""
    if df is None or df.empty:
        return None
    idx = [str(i).lower() for i in df.index]
    for k in key_candidates:
        k_low = k.lower()
        for i, name in enumerate(idx):
            if k_low == name or k_low in name:
                return pd.to_numeric(df.iloc[i], errors="coerce")
    return None

# ------------------------ Cálculo de métricas (Yahoo only) ------------------------
def compute_history_pe_yf(ticker, info, income, shares, years=10):
    """
    PE históricos 5-10y: EPS anual ~ NetIncome / Shares (aprox).
    Precio de fin de año ~ último cierre de cada año (usamos yfinance history anual).
    """
    try:
        # EPS anual aproximado
        eps_series = None
        # Net Income
        ni = series_from_df_row(income, ["Net Income", "NetIncome"])
        if ni is not None:
            # Shares: usamos shares outstanding anual si existe; si no, usar info.get("sharesOutstanding")
            so_hist = None
            if isinstance(shares, pd.DataFrame) and not shares.empty:
                # filtra annual únicamente si el DF viene con 'Yearly' / 'Annual' (estructura depende versión)
                try:
                    so_hist = shares.reset_index()
                    if "Date" in so_hist.columns and "Shares Outstanding" in so_hist.columns:
                        so_hist = so_hist[["Date","Shares Outstanding"]].dropna()
                        so_hist["Date"] = pd.to_datetime(so_hist["Date"]).dt.year
                        so_hist = so_hist.drop_duplicates("Date").set_index("Date")["Shares Outstanding"]
                except Exception:
                    so_hist = None
            if (so_hist is None or so_hist.empty) and info:
                so_hist = pd.Series({datetime.now().year: info.get("sharesOutstanding", np.nan)})
            # alinear por año
            if so_hist is not None and not so_hist.empty:
                years_idx = []
                for c in income.columns:
                    try:
                        years_idx.append(pd.to_datetime(str(c)).year)
                    except Exception:
                        years_idx.append(None)
                eps_vals = []
                years_valid = []
                for j, col in enumerate(income.columns):
                    y = years_idx[j]
                    if y is None: 
                        continue
                    ni_y = ni[col]
                    so_y = so_hist.loc[so_hist.index.get_loc(y, method="nearest")] if y in so_hist.index else so_hist.iloc[-1]
                    if pd.notna(ni_y) and pd.notna(so_y) and so_y>0:
                        eps_vals.append(float(ni_y)/float(so_y))
                        years_valid.append(y)
                if eps_vals:
                    eps_series = pd.Series(eps_vals, index=years_valid).sort_index()
        # Precios de fin de año
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
            px = price_yr[price_yr.index==y]
            if not px.empty and pd.notna(eps) and eps>0:
                pe_list.append(float(px.iloc[0])/float(eps))
        if not pe_list:
            return np.nan, np.nan
        pe_series = pd.Series(pe_list)
        pe10 = float(pe_series.tail(min(10,len(pe_series))).mean())
        pe5  = float(pe_series.tail(min(5, len(pe_series))).mean())
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
                # tendencia ~5 años: % cambio vs ~n años atrás
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
    ni  = series_from_df_row(income, ["Net Income","NetIncome"])
    # EPS: si no hay, aproximar con NI / última acción disponible (omitimos si no fiable)
    if eps is None and ni is not None:
        eps = None  # preferimos no inventar EPS sin shares anuales
    def cagr(series, yrs=5):
        if series is None or series.dropna().shape[0] < 2:
            return np.nan
        s = series.dropna().astype(float)
        first = s.iloc[max(0, len(s)-yrs-1)]
        last  = s.iloc[-1]
        years = min(yrs, len(s)-1)
        if first>0 and last>0 and years>0:
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
    e_g = growth_last(ebitda)
    r_g = growth_last(rev)
    out["op_leverage"] = float(e_g - r_g) if pd.notna(e_g) and pd.notna(r_g) else np.nan
    return out

def compute_cash_quality_and_fcf(income, cashflow):
    out = {"cfo_ni_ratio": np.nan, "fcf_margin": np.nan}
    if cashflow is None or cashflow.empty or income is None or income.empty:
        return out
    # últimas columnas
    cf_col, _ = df_latest_col(cashflow)
    inc_col, _ = df_latest_col(income)
    if cf_col is None or inc_col is None:
        return out
    cfo = float(cf_col.get("Operating Cash Flow", np.nan) or cf_col.get("Total Cash From Operating Activities", np.nan) or np.nan)
    capex = float(cf_col.get("Capital Expenditure", np.nan) or cf_col.get("Capital Expenditures", np.nan) or np.nan)
    ni = float(inc_col.get("Net Income", np.nan) or inc_col.get("NetIncome", np.nan) or np.nan)
    rev = float(inc_col.get("Total Revenue", np.nan) or inc_col.get("Revenue", np.nan) or np.nan)
    if pd.notna(cfo) and pd.notna(ni) and ni!=0:
        out["cfo_ni_ratio"] = cfo/ni
    if pd.notna(cfo) and pd.notna(capex):
        fcf = cfo + capex  # capex suele ser negativo
        if pd.notna(rev) and rev!=0:
            out["fcf_margin"] = fcf/rev
    return out

def compute_balance_ratios(income, balance, info):
    out = {"de_ratio": np.nan, "interest_coverage": np.nan, "current_ratio": np.nan,
           "p_b": np.nan, "p_fcf": np.nan, "ev_ebitda_ttm": np.nan, "roe": np.nan, "roa": np.nan}
    if balance is None or balance.empty:
        bal_col = None
    else:
        bal_col, _ = df_latest_col(balance)
    inc_col, _ = df_latest_col(income) if income is not None and not income.empty else (None, None)

    # D/E
    if bal_col is not None:
        td = float(bal_col.get("Total Debt", np.nan) or
                   ( (bal_col.get("Short Long Term Debt", np.nan) or 0) + (bal_col.get("Long Term Debt", np.nan) or 0) ))
        te = float(bal_col.get("Total Stockholder Equity", np.nan) or bal_col.get("Total Equity", np.nan) or np.nan)
        ca = float(bal_col.get("Total Current Assets", np.nan) or np.nan)
        cl = float(bal_col.get("Total Current Liabilities", np.nan) or np.nan)
        ta = float(bal_col.get("Total Assets", np.nan) or np.nan)
        out["de_ratio"] = safe_div(td, te)
        out["current_ratio"] = safe_div(ca, cl)
        out["roa"] = safe_div( (inc_col.get("Net Income", np.nan) if inc_col is not None else np.nan), ta )

    # Cobertura de intereses
    if inc_col is not None:
        ebit = float(inc_col.get("EBIT", np.nan) or inc_col.get("Operating Income", np.nan) or np.nan)
        interest_exp = inc_col.get("Interest Expense", np.nan)
        if pd.notna(interest_exp):
            interest_exp = abs(float(interest_exp))
        else:
            interest_exp = np.nan
        out["interest_coverage"] = safe_div(ebit, interest_exp)

    # ROE
    if inc_col is not None and bal_col is not None:
        ni = float(inc_col.get("Net Income", np.nan) or np.nan)
        te = float(bal_col.get("Total Stockholder Equity", np.nan) or np.nan)
        out["roe"] = safe_div(ni, te)

    # P/B, P/FCF, EV/EBITDA
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    mcap = info.get("marketCap")
    total_debt = info.get("totalDebt")
    cash = info.get("totalCash")
    ebitda_info = info.get("ebitda")

    # FCF (último año)
    if income is not None and not income.empty:
        cf = yfinance_cashflow_last = None
    # Para P/FCF necesitamos FCF último año:
    # ya se calculó en compute_cash_quality_and_fcf, pero repetimos con df_latest_col:
    pfcf = np.nan
    return_out_cf = {}
    try:
        cashflow = yf.Ticker(info.get("symbol","")).cashflow  # no fiable; lo calculamos otra vez abajo
    except Exception:
        cashflow = None

    # Mejor: reconstrúyelo a partir de balance/income llamando otra vez:
    # (lo pasamos como argumento a esta función en el main para no duplicar)
    # lo dejaremos como NaN aquí; el main seteará p_fcf usando 'fcf_margin' y revenue + market cap.

    # EV/EBITDA
    if pd.notna(mcap):
        ev = mcap + (total_debt or 0) - (cash or 0)
        if ebitda_info and ebitda_info != 0:
            out["ev_ebitda_ttm"] = ev / ebitda_info

    # P/B via marketCap / book
    if pd.notna(mcap) and bal_col is not None:
        te = float(bal_col.get("Total Stockholder Equity", np.nan) or np.nan)
        out["p_b"] = safe_div(mcap, te)

    return out

def compute_roic_wacc_proxy(income, balance, info):
    """
    ROIC ~ NOPAT / Invested Capital
    NOPAT = EBIT * (1 - tasa)
    tasa ~ Taxes / Pretax Income si disponible, si no 21%
    Invested Capital ~ Total Debt + Total Equity - Cash
    """
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
    # WACC proxy
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

def compute_p_fcf(mcap, income, cashflow):
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
    """Tendencia de acciones en circulación 3-5 años (negativa = recompras)."""
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

# ------------------------ Evaluación y scoring ------------------------
def evaluate_company(row, sector_stats):
    TH = {
        "target_upside_ratio": 1.25,  # target/price >= 1.25 (precio ≥20% por debajo del target)
        "pe_hist_discount": 0.80,     # forwardPE <= 0.8 * PE_hist_avg
        "pe_vs_sector": 0.90,         # P/E <= 0.9 * mediana sector
        "ev_ebitda_vs_sector": 0.90,  # EV/EBITDA <= 0.9 * mediana sector
        "cfo_ni_min": 0.90,           # CFO/NI >= 0.90
        "fcf_margin_min": 0.10,       # FCF margin >= 10%
        "roic_min": 0.12,             # ROIC >= 12% si no hay WACC
        "de_max": 1.00,               # D/E <= 1
        "int_cov_min": 5.0,           # Interest coverage >= 5x
        "current_ratio_min": 1.5,     # Current ratio >= 1.5
        "p_fcf_vs_sector": 0.90,      # P/FCF <= 0.9 * mediana sector
        "p_b_vs_sector": 0.90,        # P/B   <= 0.9 * mediana sector
        "peg_max": 1.0,               # PEG <= 1
        "op_leverage_min": 0.0,       # EBITDA growth - Revenue growth >= 0
    }
    checks = {}
    # 1) precio vs target
    tgt_ratio = safe_div(row.get("target_mean"), row.get("price"))
    checks["price_vs_target"] = bool(pd.notna(tgt_ratio) and tgt_ratio >= TH["target_upside_ratio"])
    # 2) forwardPE vs PE histórico
    fpe = row.get("forward_pe")
    hist_avg = nanmean_or_nan([row.get("pe_5y"), row.get("pe_10y")])
    checks["fpe_vs_hist"] = bool(pd.notna(fpe) and pd.notna(hist_avg) and fpe <= TH["pe_hist_discount"]*hist_avg)
    # 3) P/E vs sector
    sect = sector_stats.get(row.get("Sector","Unknown"), {})
    pe_now = row.get("pe_ttm")
    checks["pe_vs_sector"] = bool(pd.notna(pe_now) and pd.notna(sect.get("pe_ttm_median")) and pe_now <= TH["pe_vs_sector"]*sect["pe_ttm_median"])
    # 4) EV/EBITDA vs sector
    evx = row.get("ev_ebitda_ttm")
    checks["ev_ebitda_vs_sector"] = bool(pd.notna(evx) and pd.notna(sect.get("ev_ebitda_ttm_median")) and evx <= TH["ev_ebitda_vs_sector"]*sect["ev_ebitda_ttm_median"])
    # 5) cashflow vs deuda
    checks["cfo_ni"] = bool(pd.notna(row.get("cfo_ni_ratio")) and row["cfo_ni_ratio"] >= TH["cfo_ni_min"])
    checks["fcf_margin"] = bool(pd.notna(row.get("fcf_margin")) and row["fcf_margin"] >= TH["fcf_margin_min"])
    # ROIC vs WACC (o mínimo)
    roic = row.get("roic"); wacc = row.get("wacc_proxy")
    checks["roic_vs_wacc_or_min"] = bool( (pd.notna(roic) and pd.notna(wacc) and (roic - wacc) >= 0.02) or (pd.notna(roic) and roic >= TH["roic_min"]) )
    # márgenes/trend
    for k in ["grossProfitMargin_trend","operatingProfitMargin_trend","netProfitMargin_trend"]:
        v = row.get(k); checks[k] = bool(pd.notna(v) and v>=0)
    # apalancamiento operativo
    checks["op_leverage"] = bool(pd.notna(row.get("op_leverage")) and row["op_leverage"] >= TH["op_leverage_min"])
    # solvencia
    checks["de_ratio"] = bool(pd.notna(row.get("de_ratio")) and row["de_ratio"] <= TH["de_max"])
    checks["interest_coverage"] = bool(pd.notna(row.get("interest_coverage")) and row["interest_coverage"] >= TH["int_cov_min"])
    checks["current_ratio"] = bool(pd.notna(row.get("current_ratio")) and row["current_ratio"] >= TH["current_ratio_min"])
    # P/FCF y P/B vs sector
    for key, sect_key, th in [("p_fcf","p_fcf_median","p_fcf_vs_sector"), ("p_b","p_b_median","p_b_vs_sector")]:
        val = row.get(key); med = sect.get(sect_key)
        checks[key+"_vs_sector"] = bool(pd.notna(val) and pd.notna(med) and val <= TH[th]*med)
    # PEG
    peg = row.get("peg"); checks["peg"] = bool(pd.notna(peg) and peg <= TH["peg_max"])
    # crecimiento
    checks["rev_cagr_pos"] = bool(pd.notna(row.get("rev_cagr")) and row["rev_cagr"]>0)
    checks["eps_cagr_pos"] = bool(pd.notna(row.get("eps_cagr")) and row["eps_cagr"]>0)
    # buybacks
    sh_tr = row.get("shares_out_trend"); checks["buybacks"] = bool(pd.notna(sh_tr) and sh_tr<0)
    # insiders -> no disponible gratis; lo dejamos neutral (no penaliza)
    checks["insiders"] = True

    # scores parciales
    def rel_less(x, sect):
        if pd.isna(x) or pd.isna(sect) or sect<=0: return 0.5
        return float(np.clip(sect/x, 0, 2))/2.0
    s = sector_stats.get(row.get("Sector","Unknown"), {})
    s_val = nanmean_or_nan([
        1.0 if checks["fpe_vs_hist"] else 0.0,
        rel_less(row.get("pe_ttm"), s.get("pe_ttm_median")),
        rel_less(row.get("ev_ebitda_ttm"), s.get("ev_ebitda_ttm_median")),
        1.0 if checks["p_fcf_vs_sector"] else 0.0,
        1.0 if checks["p_b_vs_sector"] else 0.0,
        1.0 if checks["peg"] else 0.0,
    ])
    s_qlt = nanmean_or_nan([
        1.0 if checks["roic_vs_wacc_or_min"] else 0.0,
        1.0 if checks["cfo_ni"] else 0.0,
        1.0 if checks["fcf_margin"] else 0.0,
        1.0 if checks["op_leverage"] else 0.0,
        1.0 if checks["grossProfitMargin_trend"] else 0.0,
        1.0 if checks["operatingProfitMargin_trend"] else 0.0,
        1.0 if checks["netProfitMargin_trend"] else 0.0,
    ])
    s_sol = nanmean_or_nan([1.0 if checks["de_ratio"] else 0.0,
                            1.0 if checks["interest_coverage"] else 0.0,
                            1.0 if checks["current_ratio"] else 0.0])
    s_act = nanmean_or_nan([1.0 if checks["buybacks"] else 0.0, 1.0])  # insiders=neutral
    s_gro = nanmean_or_nan([1.0 if checks["rev_cagr_pos"] else 0.0,
                            1.0 if checks["eps_cagr_pos"] else 0.0])

    weights = {"valoracion":0.30, "calidad":0.35, "solvencia":0.20, "accionista":0.10, "crecimiento":0.05}
    score = (weights["valoracion"]*(s_val or 0.0) +
             weights["calidad"]*(s_qlt or 0.0) +
             weights["solvencia"]*(s_sol or 0.0) +
             weights["accionista"]*(s_act or 0.0) +
             weights["crecimiento"]*(s_gro or 0.0))

    hard_ok = checks["price_vs_target"] and sum([checks["fpe_vs_hist"], checks["pe_vs_sector"], checks["ev_ebitda_vs_sector"]]) >= 2
    if not hard_ok:
        score *= 0.5
    return score, checks

# ------------------------ Motor ------------------------
def main(max_names=500, top_k=10):
    print("Iniciando el proceso de screening...")
    sp = load_sp500_constituents()
    if "Ticker" not in sp.columns:
        sp["Ticker"] = sp.iloc[:,0]
    if "Sector" not in sp.columns:
        sp["Sector"] = np.nan
    tickers = sp["Ticker"].astype(str).str.replace(".", "-", regex=False).tolist()
    if max_names:
        tickers = tickers[:max_names]

    rows = []
    print(f"Procesando {len(tickers)} tickers...")
    for i, t in enumerate(tickers, 1):
        print(f"Procesando ticker {i}/{len(tickers)}: {t}")
        try:
            info = yf_safe_info(t) or {}
            price, target = yf_price_and_target(t)
            fin = yf_financials(t)
            income = fin["income"]
            balance = fin["balance"]
            cashflow = fin["cashflow"]
            shares = fin["shares"]

            print(f"  - Obteniendo datos básicos para {t}")
            # Básicos
            sector = info.get("sector") or sp.loc[sp["Ticker"]==t, "Sector"].fillna("Unknown").values[0] if "sector" in info else sp.loc[sp["Ticker"]==t, "Sector"].fillna("Unknown").values[0]
            name = info.get("longName") or info.get("shortName") or sp.loc[sp["Ticker"]==t, "Name"].values[0] if "Name" in sp.columns else None
            pe_ttm = info.get("trailingPE")
            forward_pe = info.get("forwardPE")

            print(f"  - Calculando P/E histórico para {t}")
            # P/E históricos
            pe10, pe5 = compute_history_pe_yf(t, info, income, shares, years=10)

            print(f"  - Calculando márgenes y tendencias para {t}")
            # Márgenes y tendencias
            margins = compute_margins_and_trends(income)

            print(f"  - Calculando crecimiento para {t}")
            # Crecimientos
            growth = compute_growth(income)

            print(f"  - Calculando apalancamiento operativo para {t}")
            # Apalancamiento operativo
            op_lev = compute_operating_leverage(income)

            print(f"  - Calculando calidad del cashflow y FCF para {t}")
            # Cash quality + FCF margin
            cashq = compute_cash_quality_and_fcf(income, cashflow)

            print(f"  - Calculando ratios de balance para {t}")
            # Balance ratios + valuation
            balrat = compute_balance_ratios(income, balance, info)

            print(f"  - Calculando P/FCF para {t}")
            # P/FCF (market cap / FCF último año)
            p_fcf = compute_p_fcf(info.get("marketCap"), income, cashflow)

            print(f"  - Calculando ROIC y WACC para {t}")
            # ROIC y WACC proxy
            roic_wacc = compute_roic_wacc_proxy(income, balance, info)

            # PEG con EPS growth si hay
            peg = np.nan
            if pd.notna(forward_pe) and pd.notna(growth.get("eps_cagr")) and growth["eps_cagr"]>0:
                # PEG = P/E / growth%. Si growth en proporción, lo paso a %
                g = growth["eps_cagr"]
                peg = forward_pe / (g*100 if g<1 else g)

            print(f"  - Calculando recompras para {t}")
            # Buybacks
            sh_trend = compute_buybacks(shares)

            row = {
                "Ticker": t, "Name": name, "Sector": sector if sector else "Unknown",
                "price": float(price) if pd.notna(price) else np.nan,
                "target_mean": float(target) if pd.notna(target) else np.nan,
                "pe_ttm": float(pe_ttm) if pd.notna(pe_ttm) else np.nan,
                "forward_pe": float(forward_pe) if pd.notna(forward_pe) else np.nan,
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
            rows.append(row)
        except Exception as e:
            # continúa con el siguiente
            continue

    print("Todos los tickers han sido procesados. Creando DataFrame...")
    df = pd.DataFrame(rows)
    # Asegura columnas
    ensure_cols = [
        "price","target_mean","pe_ttm","forward_pe","pe_5y","pe_10y","ev_ebitda_ttm","p_fcf","p_b",
        "de_ratio","interest_coverage","current_ratio","cfo_ni_ratio","fcf_margin","roic","wacc_proxy",
        "op_leverage","grossProfitMargin_last","operatingProfitMargin_last","netProfitMargin_last",
        "grossProfitMargin_trend","operatingProfitMargin_trend","netProfitMargin_trend","peg","rev_cagr",
        "eps_cagr","shares_out_trend","Name","Sector"
    ]
    for c in ensure_cols:
        if c not in df.columns: df[c] = np.nan

    # Rellena márgenes (si no vinieron)
    # (ya se añaden en margins arriba; por seguridad)
    # Estadísticas sectoriales (medianas)
    print("Calculando estadísticas por sector...")
    sector_stats = {}
    for s, g in df.groupby("Sector"):
        sector_stats[s] = {
            "pe_ttm_median": np.nanmedian(g["pe_ttm"].values) if "pe_ttm" in g else np.nan,
            "ev_ebitda_ttm_median": np.nanmedian(g["ev_ebitda_ttm"].values) if "ev_ebitda_ttm" in g else np.nan,
            "p_fcf_median": np.nanmedian(g["p_fcf"].values) if "p_fcf" in g else np.nan,
            "p_b_median": np.nanmedian(g["p_b"].values) if "p_b" in g else np.nan,
            "de_ratio_median": np.nanmedian(g["de_ratio"].values) if "de_ratio" in g else np.nan,
            "current_ratio_median": np.nanmedian(g["current_ratio"].values) if "current_ratio" in g else np.nan,
            "interest_coverage_median": np.nanmedian(g["interest_coverage"].values) if "interest_coverage" in g else np.nan,
            "fcf_margin_median": np.nanmedian(g["fcf_margin"].values) if "fcf_margin" in g else np.nan,
            "cfo_ni_ratio_median": np.nanmedian(g["cfo_ni_ratio"].values) if "cfo_ni_ratio" in g else np.nan,
        }

    # Scoring
    print("Realizando scoring y evaluación de compañías...")
    scores, checks_all = [], []
    for _, r in df.iterrows():
        sc, ch = evaluate_company(r.to_dict(), sector_stats)
        scores.append(sc); checks_all.append(ch)
    df["score"] = scores
    df["checks"] = checks_all
    df = df.sort_values("score", ascending=False)

    # Informe Top-K
    print("Generando informe Top-K...")
    top_k = min(10, len(df))
    top = df.head(top_k).copy()

    def _yn(b): return "✅" if b else "❌"
    def _fmt(val, nd=2): return f"{val:.{nd}f}" if pd.notna(val) else "n/d"
    def _pct(val): return f"{val*100:.1f}%" if pd.notna(val) else "n/d"

    print("\n" + "="*95)
    print(f"TOP {top_k} S&P 500 por potencial fundamental (mediano plazo) — {datetime.now().date()}")
    print("="*95 + "\n")

    for _, r in top.iterrows():
        ch = r["checks"]; sect = sector_stats.get(r.get("Sector","Unknown"), {})
        price = r.get("price", np.nan); tgt = r.get("target_mean", np.nan)
        tgr = (tgt/price) if (pd.notna(tgt) and pd.notna(price) and price>0) else np.nan
        pe_hist_avg = nanmean_or_nan([r.get("pe_5y", np.nan), r.get("pe_10y", np.nan)])

        print(f"▶ {r.get('Ticker','?'):<8}  {str(r.get('Name','')).strip()[:60]}  |  Sector: {r.get('Sector','Unknown')}")
        print(f"   Precio actual: {_fmt(price)}  | Objetivo (media): {_fmt(tgt)}")
        print(f"   Debería: target/price ≥ 1.25  | Está en: {_fmt(tgr)} -> {_yn(ch['price_vs_target'])}")

        fpe = r.get("forward_pe", np.nan)
        if pd.notna(fpe) and pd.notna(pe_hist_avg):
            print(f"   Forward P/E vs media 5-10y: debería ≤ 0.80×hist | {fpe:.2f} vs {pe_hist_avg:.2f} -> {_yn(ch['fpe_vs_hist'])}")
        elif pd.notna(fpe):
            print(f"   Forward P/E (sin hist confiable): {fpe:.2f} -> ⚠️ datos históricos insuficientes")
        else:
            print("   Forward P/E: n/d -> ⚠️")

        pe_now = r.get("pe_ttm", np.nan); pe_sct = sect.get("pe_ttm_median", np.nan)
        if pd.notna(pe_now) and pd.notna(pe_sct):
            print(f"   P/E vs sector (mediana {pe_sct:.2f}): debería ≤ 0.90×sector | {pe_now:.2f} -> {_yn(ch['pe_vs_sector'])}")

        ev_eb = r.get("ev_ebitda_ttm", np.nan); ev_s = sect.get("ev_ebitda_ttm_median", np.nan)
        if pd.notna(ev_eb) and pd.notna(ev_s):
            print(f"   EV/EBITDA vs sector (med {ev_s:.2f}): debería ≤ 0.90×sector | {ev_eb:.2f} -> {_yn(ch['ev_ebitda_vs_sector'])}")

        print(f"   CFO/NI: debería ≥ 0.90 | {_fmt(r.get('cfo_ni_ratio'))} -> {_yn(ch['cfo_ni'])}")
        print(f"   FCF margin: debería ≥ 10% | {_pct(r.get('fcf_margin'))} -> {_yn(ch['fcf_margin'])}")

        print(f"   ROIC vs WACC/min: ROIC {_pct(r.get('roic'))} | WACC~ {_pct(r.get('wacc_proxy'))} -> {_yn(ch['roic_vs_wacc_or_min'])}")

        for label, key_last, key_trend in [
            ("Margen Bruto","grossProfitMargin_last","grossProfitMargin_trend"),
            ("Margen Operativo","operatingProfitMargin_last","operatingProfitMargin_trend"),
            ("Margen Neto","netProfitMargin_last","netProfitMargin_trend"),
        ]:
            last = r.get(key_last, np.nan); tr = r.get(key_trend, np.nan)
            if pd.notna(last):
                sign = "+" if pd.notna(tr) and tr>=0 else ""
                trend_txt = f"{sign}{tr*100:.1f}%" if pd.notna(tr) else "n/d"
                print(f"   {label}: {_pct(last)}  | Tendencia ~5a: {trend_txt} -> {'✅' if (pd.notna(tr) and tr>=0) else ('⚠️' if pd.isna(tr) else '❌')}")

        if pd.notna(r.get("de_ratio")):
            print(f"   D/E: debería ≤ 1.0 | {r['de_ratio']:.2f} -> {_yn(ch['de_ratio'])}")
        if pd.notna(r.get("interest_coverage")):
            print(f"   Cobertura intereses: ≥ 5x | {r['interest_coverage']:.1f}x -> {_yn(ch['interest_coverage'])}")
        if pd.notna(r.get("current_ratio")):
            print(f"   Current ratio: ≥ 1.5 | {r['current_ratio']:.2f} -> {_yn(ch['current_ratio'])}")

        p_fcf = r.get("p_fcf", np.nan); p_fcf_s = sect.get("p_fcf_median", np.nan)
        if pd.notna(p_fcf) and pd.notna(p_fcf_s):
            print(f"   P/FCF vs sector (med {p_fcf_s:.2f}): ≤ 0.90×sector | {p_fcf:.2f} -> {_yn(ch['p_fcf_vs_sector'])}")
        p_b = r.get("p_b", np.nan); p_b_s = sect.get("p_b_median", np.nan)
        if pd.notna(p_b) and pd.notna(p_b_s):
            print(f"   P/B vs sector (med {p_b_s:.2f}): ≤ 0.90×sector | {p_b:.2f} -> {_yn(ch['p_b_vs_sector'])}")

        if pd.notna(r.get("peg")):
            print(f"   PEG: ≤ 1.0 | {r['peg']:.2f} -> {_yn(ch['peg'])}")
        if pd.notna(r.get("rev_cagr")):
            print(f"   CAGR Ingresos (3-5a aprox): {_pct(r.get('rev_cagr'))} -> {_yn(ch['rev_cagr_pos'])}")
        if pd.notna(r.get("eps_cagr")):
            print(f"   CAGR EPS (3-5a aprox): {_pct(r.get('eps_cagr'))} -> {_yn(ch['eps_cagr_pos'])}")

        sh_tr = r.get("shares_out_trend", np.nan)
        if pd.notna(sh_tr):
            print(f"   Tendencia acciones en circ.: {'↓ recompras' if sh_tr<0 else '↑ emisión'} ({sh_tr*100:.1f}% ~5a) -> {_yn(ch['buybacks'])}")

        print(f"   ▶ Score compuesto: {_fmt(r.get('score', np.nan), 3)}")
        print("-"*95)

    # Export CSV completo (no solo top)
    outcols = [c for c in df.columns if c != "checks"]
    df[outcols].to_csv("sp500_top10_fundamental_screen.csv", index=False)
    print("\nArchivo guardado: sp500_top10_fundamental_screen.csv")
    print("\nNotas:")
    print(" - Todo sale de Yahoo (yfinance): gratis, pero algunas partidas pueden faltar para ciertos tickers.")
    print(" - Si puedes aportar un CSV local del S&P 500 reciente, úsalo para máxima robustez.")
    print(" - Los umbrales están pensados para mediano plazo; ajusta TH en evaluate_company() si quieres.")
    print(" - ROIC y WACC son aproximados (contabilidad simplificada). PEG depende de EPS CAGR disponible.")
    print(" - Las señales de insiders no se incluyen por falta de fuente gratuita consolidada (neutral en score).")
    print("Proceso de screening finalizado.")

if __name__ == "__main__":
    try:
        main(max_names=500, top_k=10)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    except Exception as e:
        print("Error durante la ejecución:\n", traceback.format_exc())
