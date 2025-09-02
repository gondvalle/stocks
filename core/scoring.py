# /sp500_screener/core/scoring.py
from __future__ import annotations
import numpy as np
from .utils import safe_div, nanmean_or_nan

THRESHOLDS = {
    "target_upside_ratio": 1.25,  # target/price >= 1.25
    "pe_hist_discount": 0.80,     # forwardPE <= 0.8 * PE_hist_avg
    "pe_vs_sector": 0.90,         # P/E <= 0.9 * mediana sector
    "ev_ebitda_vs_sector": 0.90,  # EV/EBITDA <= 0.9 * mediana sector
    "cfo_ni_min": 0.90,           # CFO/NI >= 0.90
    "fcf_margin_min": 0.10,       # FCF margin >= 10%
    "roic_min": 0.12,             # ROIC >= 12%
    "de_max": 1.00,               # D/E <= 1
    "int_cov_min": 5.0,           # Interest coverage >= 5x
    "current_ratio_min": 1.5,     # Current ratio >= 1.5
    "p_fcf_vs_sector": 0.90,      # P/FCF <= 0.9 * mediana sector
    "p_b_vs_sector": 0.90,        # P/B <= 0.9 * mediana sector
    "peg_max": 1.0,               # PEG <= 1
    "op_leverage_min": 0.0,       # EBITDA growth - Revenue growth >= 0
}

def evaluate_company(row: dict, sector_stats: dict) -> tuple[float, dict]:
    checks = {}
    tgt_ratio = safe_div(row.get("target_mean"), row.get("price"))
    checks["price_vs_target"] = bool((tgt_ratio is not None) and (tgt_ratio >= THRESHOLDS["target_upside_ratio"]))

    fpe = row.get("forward_pe")
    pe_hist_avg = nanmean_or_nan([row.get("pe_5y"), row.get("pe_10y")])
    checks["fpe_vs_hist"] = bool(fpe is not None and pe_hist_avg is not None and fpe <= THRESHOLDS["pe_hist_discount"] * pe_hist_avg)

    sect = sector_stats.get(row.get("Sector","Unknown"), {})
    pe_now = row.get("pe_ttm")
    checks["pe_vs_sector"] = bool(pe_now is not None and sect.get("pe_ttm_median") is not None and pe_now <= THRESHOLDS["pe_vs_sector"]*sect["pe_ttm_median"])

    evx = row.get("ev_ebitda_ttm")
    checks["ev_ebitda_vs_sector"] = bool(evx is not None and sect.get("ev_ebitda_ttm_median") is not None and evx <= THRESHOLDS["ev_ebitda_vs_sector"]*sect["ev_ebitda_ttm_median"])

    checks["cfo_ni"] = bool(row.get("cfo_ni_ratio") is not None and row["cfo_ni_ratio"] >= THRESHOLDS["cfo_ni_min"])
    checks["fcf_margin"] = bool(row.get("fcf_margin") is not None and row["fcf_margin"] >= THRESHOLDS["fcf_margin_min"])

    roic = row.get("roic"); wacc = row.get("wacc_proxy")
    checks["roic_vs_wacc_or_min"] = bool((roic is not None and wacc is not None and (roic - wacc) >= 0.02) or (roic is not None and roic >= THRESHOLDS["roic_min"]))

    for k in ["grossProfitMargin_trend","operatingProfitMargin_trend","netProfitMargin_trend"]:
        v = row.get(k); checks[k] = bool(v is not None and v >= 0)

    checks["op_leverage"] = bool(row.get("op_leverage") is not None and row["op_leverage"] >= THRESHOLDS["op_leverage_min"])

    checks["de_ratio"] = bool(row.get("de_ratio") is not None and row["de_ratio"] <= THRESHOLDS["de_max"])
    checks["interest_coverage"] = bool(row.get("interest_coverage") is not None and row["interest_coverage"] >= THRESHOLDS["int_cov_min"])
    checks["current_ratio"] = bool(row.get("current_ratio") is not None and row["current_ratio"] >= THRESHOLDS["current_ratio_min"])

    for key, sect_key, th in [("p_fcf","p_fcf_median","p_fcf_vs_sector"), ("p_b","p_b_median","p_b_vs_sector")]:
        val = row.get(key); med = sect.get(sect_key)
        checks[key+"_vs_sector"] = bool(val is not None and med is not None and val <= THRESHOLDS[th]*med)

    peg = row.get("peg"); checks["peg"] = bool(peg is not None and peg <= THRESHOLDS["peg_max"])

    checks["rev_cagr_pos"] = bool(row.get("rev_cagr") is not None and row["rev_cagr"] > 0)
    checks["eps_cagr_pos"] = bool(row.get("eps_cagr") is not None and row["eps_cagr"] > 0)

    sh_tr = row.get("shares_out_trend"); checks["buybacks"] = bool(sh_tr is not None and sh_tr < 0)
    checks["insiders"] = True  # neutral

    def rel_less(x, sect_val):
        if x is None or sect_val is None or sect_val <= 0: return 0.5
        return float(np.clip(sect_val/x, 0, 2))/2.0

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
    s_sol = nanmean_or_nan([
        1.0 if checks["de_ratio"] else 0.0,
        1.0 if checks["interest_coverage"] else 0.0,
        1.0 if checks["current_ratio"] else 0.0,
    ])
    s_act = nanmean_or_nan([1.0 if checks["buybacks"] else 0.0, 1.0])
    s_gro = nanmean_or_nan([1.0 if checks["rev_cagr_pos"] else 0.0, 1.0 if checks["eps_cagr_pos"] else 0.0])

    weights = {"valoracion":0.30, "calidad":0.35, "solvencia":0.20, "accionista":0.10, "crecimiento":0.05}
    score = (weights["valoracion"]*(s_val or 0.0) +
             weights["calidad"]*(s_qlt or 0.0) +
             weights["solvencia"]*(s_sol or 0.0) +
             weights["accionista"]*(s_act or 0.0) +
             weights["crecimiento"]*(s_gro or 0.0))

    hard_ok = checks["price_vs_target"] and sum([checks["fpe_vs_hist"], checks["pe_vs_sector"], checks["ev_ebitda_vs_sector"]]) >= 2
    if not hard_ok:
        score *= 0.5
    return float(score), checks
