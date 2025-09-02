# /sp500_screener/ui/pages/stock_detail.py
import streamlit as st
import numpy as np
import pandas as pd

from core.fetch import yf_price_and_target, yf_financials, get_price_history_close
from core.metrics import (
    compute_history_pe_yf, compute_margins_and_trends, compute_growth,
    compute_operating_leverage, compute_cash_quality_and_fcf,
    compute_balance_ratios, compute_roic_wacc_proxy, compute_p_fcf, compute_buybacks
)
from core.scoring import evaluate_company
from core.sectors import sector_medians
from core.reporting import get_or_make_daily_scored
from ui.components import show_check_line
from ui.charts import (
    price_timeseries,
    bar_forwardpe_vs_hist,
    boxplot_sector_with_point,   # ya lo tienes y lo usamos también
    boxplot_sector_metric,       # nuevo helper
    scatter_cfo_ni_vs_de,
    scatter_fcf_margin_vs_de,
)
from core.sectors import sector_medians
from core.reporting import get_or_make_daily_scored


def render(ticker: str | None = None):
    st.header("Detalle de Acción")

    # Ticker desde session/query o input
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

    # Carga el dataframe del día y sus medianas de sector
    df_today = get_or_make_daily_scored(force_refresh=False)
    meds_today = sector_medians(df_today, exclude_ticker=t)

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
            "Ticker": t,
            "Name": info.get("longName") or info.get("shortName") or t,
            "Sector": info.get("sector") or "Unknown",
            "price": price,
            "target_mean": target,
            "pe_ttm": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pe_10y": pe10, "pe_5y": pe5,
            "ev_ebitda_ttm": balrat.get("ev_ebitda_ttm"),
            "p_fcf": p_fcf, "p_b": balrat.get("p_b"),
            "de_ratio": balrat.get("de_ratio"),
            "interest_coverage": balrat.get("interest_coverage"),
            "current_ratio": balrat.get("current_ratio"),
            "cfo_ni_ratio": cashq.get("cfo_ni_ratio"),
            "fcf_margin": cashq.get("fcf_margin"),
            "roic": roic_wacc.get("roic"),
            "wacc_proxy": roic_wacc.get("wacc_proxy"),
            "rev_cagr": growth.get("rev_cagr"),
            "eps_cagr": growth.get("eps_cagr"),
            "op_leverage": op_lev.get("op_leverage"),
            "shares_out_trend": sh_tr,
            "beta": info.get("beta"),
        }
        # PEG (misma lógica que el motor)
        if info.get("forwardPE") and growth.get("eps_cagr") and growth["eps_cagr"] > 0:
            g = growth["eps_cagr"]
            row["peg"] = info["forwardPE"] / (g*100 if g < 1 else g)
        else:
            row["peg"] = np.nan

        # Score/Checks con medianas reales del día
        score_calc, checks = evaluate_company(row, meds_today)

        # Si el ticker ya está en el DF del día, muestra su score "oficial" del día
        score_today = df_today.loc[df_today["Ticker"] == t, "score"]
        if not score_today.empty and np.isfinite(score_today.iloc[0]):
            score = float(score_today.iloc[0])
        else:
            score = float(score_calc)

     # --- NUEVA SECCIÓN: Comparativas visuales ---
    st.subheader("Comparativas visuales frente a su sector")

    sector = row["Sector"]
    label = row["Ticker"]

    # 1) Forward P/E vs histórico + referencia sector (P/E TTM mediana)
    st.plotly_chart(bar_forwardpe_vs_hist(row, meds_today), use_container_width=True)

    # 2) P/E TTM vs sector (boxplot + punto)
    st.plotly_chart(
        boxplot_sector_with_point(
            df_today, sector=sector, column="pe_ttm",
            point_value=row.get("pe_ttm", np.nan), label=label
        ),
        use_container_width=True
    )

    # 3) EV/EBITDA vs sector (boxplot + punto)
    st.plotly_chart(
        boxplot_sector_with_point(
            df_today, sector=sector, column="ev_ebitda_ttm",
            point_value=row.get("ev_ebitda_ttm", np.nan), label=label
        ),
        use_container_width=True
    )

    # 4) Cashflow vs deuda: CFO/NI vs D/E (dispersión con la empresa resaltada)
    st.plotly_chart(
        scatter_cfo_ni_vs_de(df_today, sector=sector, ticker=label),
        use_container_width=True
    )

    # 5) Alternativa/extra: FCF margin vs D/E (otra vista de caja vs deuda)
    st.plotly_chart(
        scatter_fcf_margin_vs_de(df_today, sector=sector, ticker=label),
        use_container_width=True
    )

    c0, c1, c2, c3 = st.columns([1,1,1,1])
    c0.metric("Precio", f"{price:.2f}" if price == price else "n/d")
    c1.metric("Target (mean)", f"{target:.2f}" if target == target else "n/d")
    c2.metric("Upside", f"{(target/price-1)*100:.1f}%" if (price and target and price > 0) else "n/d")
    c3.metric("Score (día)", f"{score:.3f}")

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
    c1_, c2_, c3_ = st.columns(3)
    c1_.metric("Bruto (últ.)", f"{row.get('grossProfitMargin_last', np.nan)*100:.1f}%" if row.get('grossProfitMargin_last')==row.get('grossProfitMargin_last') else "n/d")
    c1_.metric("Bruto (trend)", f"{row.get('grossProfitMargin_trend', np.nan)*100:+.1f}%" if row.get('grossProfitMargin_trend')==row.get('grossProfitMargin_trend') else "n/d")
    c2_.metric("Operativo (últ.)", f"{row.get('operatingProfitMargin_last', np.nan)*100:.1f}%" if row.get('operatingProfitMargin_last')==row.get('operatingProfitMargin_last') else "n/d")
    c2_.metric("Operativo (trend)", f"{row.get('operatingProfitMargin_trend', np.nan)*100:+.1f}%" if row.get('operatingProfitMargin_trend')==row.get('operatingProfitMargin_trend') else "n/d")
    c3_.metric("Neto (últ.)", f"{row.get('netProfitMargin_last', np.nan)*100:.1f}%" if row.get('netProfitMargin_last')==row.get('netProfitMargin_last') else "n/d")
    c3_.metric("Neto (trend)", f"{row.get('netProfitMargin_trend', np.nan)*100:+.1f}%" if row.get('netProfitMargin_trend')==row.get('netProfitMargin_trend') else "n/d")

    st.caption("Crecimiento (~5 años):")
    c4, c5 = st.columns(2)
    c4.metric("CAGR Ingresos", f"{(row.get('rev_cagr') or np.nan)*100:.1f}%" if row.get('rev_cagr')==row.get('rev_cagr') else "n/d")
    c5.metric("CAGR EPS", f"{(row.get('eps_cagr') or np.nan)*100:.1f}%" if row.get('eps_cagr')==row.get('eps_cagr') else "n/d")

    st.caption("Otras señales:")
    c6, c7 = st.columns(2)
    c6.metric("Apalancamiento operativo", f"{(row.get('op_leverage') or 0)*100:+.1f}%" if row.get('op_leverage')==row.get('op_leverage') else "n/d")
    c7.metric("Tendencia acciones (≈ recompras <0)", f"{(row.get('shares_out_trend') or 0)*100:+.1f}%" if row.get('shares_out_trend')==row.get('shares_out_trend') else "n/d")

    st.subheader("Precio")
    hist = get_price_history_close(t)
    st.plotly_chart(price_timeseries(hist, title=f"Precio histórico de {t}"), use_container_width=True)
