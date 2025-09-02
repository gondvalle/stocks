# /sp500_screener/ui/pages/stock_detail.py
import streamlit as st
import numpy as np
import pandas as pd
from core.fetch import yf_price_and_target, yf_financials
from core.metrics import compute_history_pe_yf, compute_margins_and_trends, compute_growth, compute_operating_leverage, compute_cash_quality_and_fcf, compute_balance_ratios, compute_roic_wacc_proxy, compute_p_fcf, compute_buybacks
from ui.components import show_check_line
from core.scoring import evaluate_company
from core.sectors import sector_medians

def render(ticker: str | None = None):
    st.header("Detalle de Acción")

    qp_t = st.session_state.get("selected_ticker") or st.query_params.get("ticker") or ticker
    t = st.text_input("Ticker", value=qp_t or "")
    go = st.button("Cargar")
    if (not t) and (not go):
        st.info("Introduce un ticker (ej. AAPL) y pulsa **Cargar**. También puedes llegar aquí desde el Screener/Dashboard.")
        return
    if not t and go:
        st.warning("Ticker vacío.")
        return
    if not go and qp_t:
        t = qp_t

    with st.spinner(f"Cargando {t}..."):
        price, target = yf_price_and_target(t)
        fin = yf_financials(t)
        income, balance, cashflow, shares, info = fin["income"], fin["balance"], fin["cashflow"], fin["shares"], fin["info"]

        pe10, pe5 = compute_history_pe_yf(t, info, income, shares)
        margins = compute_margins_and_trends(income)
        growth = compute_growth(income)
        op_lev = compute_operating_leverage(income)
        cashq = compute_cash_quality_and_fcf(income, cashflow)
        balrat = compute_balance_ratios(income, balance, info)
        roic_wacc = compute_roic_wacc_proxy(income, balance, info)
        p_fcf = compute_p_fcf(info.get("marketCap"), income, cashflow)
        sh_tr = compute_buybacks(shares)

        row = {
            "Ticker": t, "Name": info.get("longName") or info.get("shortName") or t, "Sector": info.get("sector") or "Unknown",
            "price": price, "target_mean": target, "pe_ttm": info.get("trailingPE"), "forward_pe": info.get("forwardPE"),
            "pe_10y": pe10, "pe_5y": pe5, "ev_ebitda_ttm": balrat.get("ev_ebitda_ttm"), "p_fcf": p_fcf, "p_b": balrat.get("p_b"),
            "de_ratio": balrat.get("de_ratio"), "interest_coverage": balrat.get("interest_coverage"), "current_ratio": balrat.get("current_ratio"),
            "cfo_ni_ratio": cashq.get("cfo_ni_ratio"), "fcf_margin": cashq.get("fcf_margin"), "roic": roic_wacc.get("roic"),
            "wacc_proxy": roic_wacc.get("wacc_proxy"), "rev_cagr": growth.get("rev_cagr"), "eps_cagr": growth.get("eps_cagr"),
            "op_leverage": op_lev.get("op_leverage"), "shares_out_trend": sh_tr,
            "grossProfitMargin_trend": margins.get("grossProfitMargin_trend"), "operatingProfitMargin_trend": margins.get("operatingProfitMargin_trend"),
            "netProfitMargin_trend": margins.get("netProfitMargin_trend"), "peg": ( (info.get("forwardPE") or np.nan) / ( (growth.get("eps_cagr")*100) if (growth.get("eps_cagr") is not None and growth["eps_cagr"]<1) else (growth.get("eps_cagr") or np.nan) ) ) if (info.get("forwardPE") and growth.get("eps_cagr") and growth["eps_cagr"]>0) else np.nan
        }

        # sector stats de esta única fila (no muy informativo), pero construimos a mano:
        sect_stats = { row["Sector"]: {
            "pe_ttm_median": row["pe_ttm"], "ev_ebitda_ttm_median": row["ev_ebitda_ttm"],
            "p_fcf_median": row["p_fcf"], "p_b_median": row["p_b"],
        } }
        score, checks = evaluate_company(row, sect_stats)

    c0, c1, c2, c3 = st.columns([1,1,1,1])
    c0.metric("Precio", f"{price:.2f}" if price==price else "n/d")
    c1.metric("Target (mean)", f"{target:.2f}" if target==target else "n/d")
    c2.metric("Upside", f"{(target/price-1)*100:.1f}%" if (price and target and price>0) else "n/d")
    c3.metric("Score", f"{score:.3f}")

    st.subheader(f"{row['Ticker']} · {row['Name']}  —  Sector: {row['Sector']}")
    left, right = st.columns([1,1])

    with left:
        show_check_line("Forward P/E", row.get("forward_pe"), checks.get("fpe_vs_hist"), "{:.2f}")
        show_check_line("P/E TTM", row.get("pe_ttm"), checks.get("pe_vs_sector"), "{:.2f}")
        show_check_line("EV/EBITDA TTM", row.get("ev_ebitda_ttm"), checks.get("ev_ebitda_vs_sector"), "{:.2f}")
        show_check_line("P/FCF", row.get("p_fcf"), checks.get("p_fcf_vs_sector"), "{:.2f}")
        show_check_line("P/B", row.get("p_b"), checks.get("p_b_vs_sector"), "{:.2f}")
        show_check_line("PEG", row.get("peg"), checks.get("peg"), "{:.2f}")

    with right:
        show_check_line("CFO/NI", row.get("cfo_ni_ratio"), checks.get("cfo_ni"), "{:.2f}")
        show_check_line("FCF margin", row.get("fcf_margin"), checks.get("fcf_margin"), "{:.2%}")
        show_check_line("ROIC", row.get("roic"), None, "{:.2%}")
        show_check_line("WACC (proxy)", row.get("wacc_proxy"), None, "{:.2%}")
        show_check_line("D/E", row.get("de_ratio"), checks.get("de_ratio"), "{:.2f}")
        show_check_line("Interest coverage", row.get("interest_coverage"), checks.get("interest_coverage"), "{:.1f}x")
        show_check_line("Current ratio", row.get("current_ratio"), checks.get("current_ratio"), "{:.2f}")

    st.caption("Tendencias de márgenes (~5 años):")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bruto (últ.)", f"{margins.get('grossProfitMargin_last', np.nan)*100:.1f}%" if margins.get('grossProfitMargin_last')==margins.get('grossProfitMargin_last') else "n/d")
    c1.metric("Bruto (trend)", f"{margins.get('grossProfitMargin_trend', np.nan)*100:+.1f}%" if margins.get('grossProfitMargin_trend')==margins.get('grossProfitMargin_trend') else "n/d")
    c2.metric("Operativo (últ.)", f"{margins.get('operatingProfitMargin_last', np.nan)*100:.1f}%" if margins.get('operatingProfitMargin_last')==margins.get('operatingProfitMargin_last') else "n/d")
    c2.metric("Operativo (trend)", f"{margins.get('operatingProfitMargin_trend', np.nan)*100:+.1f}%" if margins.get('operatingProfitMargin_trend')==margins.get('operatingProfitMargin_trend') else "n/d")
    c3.metric("Neto (últ.)", f"{margins.get('netProfitMargin_last', np.nan)*100:.1f}%" if margins.get('netProfitMargin_last')==margins.get('netProfitMargin_last') else "n/d")
    c3.metric("Neto (trend)", f"{margins.get('netProfitMargin_trend', np.nan)*100:+.1f}%" if margins.get('netProfitMargin_trend')==margins.get('netProfitMargin_trend') else "n/d")

    st.caption("Crecimiento (~5 años):")
    c4, c5 = st.columns(2)
    c4.metric("CAGR Ingresos", f"{(row.get('rev_cagr') or np.nan)*100:.1f}%" if row.get('rev_cagr')==row.get('rev_cagr') else "n/d")
    c5.metric("CAGR EPS", f"{(row.get('eps_cagr') or np.nan)*100:.1f}%" if row.get('eps_cagr')==row.get('eps_cagr') else "n/d")

    st.caption("Otras señales:")
    c6, c7 = st.columns(2)
    c6.metric("Apalancamiento operativo", f"{(row.get('op_leverage') or 0)*100:+.1f}%" if row.get('op_leverage')==row.get('op_leverage') else "n/d")
    c7.metric("Tendencia acciones (≈ recompras <0)", f"{(row.get('shares_out_trend') or 0)*100:+.1f}%" if row.get('shares_out_trend')==row.get('shares_out_trend') else "n/d")
