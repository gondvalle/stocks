from __future__ import annotations
import streamlit as st
import pandas as pd
from ui.charts import boxplot_sector_with_point, price_timeseries
from core.reporting import get_or_make_daily_scored
from core.fetch import get_price_history_close

def render():
    st.header("Comparativas por sector (empresa a empresa)")

    df = get_or_make_daily_scored(force_refresh=False)
    all_tickers = df["Ticker"].tolist()

    c1, c2 = st.columns([2,1])
    with c1:
        sel = st.multiselect("Selecciona compañías para comparar (una a una):", all_tickers, default=all_tickers[:3])
    with c2:
        metrics = ["pe_ttm","forward_pe","ev_ebitda_ttm","p_fcf","p_b","fcf_margin","cfo_ni_ratio","roic","rev_cagr","eps_cagr"]
        metric = st.selectbox("Métrica a resaltar vs sector:", metrics, index=0)

    if not sel:
        st.info("Elige al menos una compañía.")
        return

    for t in sel:
        sub = df[df["Ticker"] == t]
        if sub.empty:
            continue
        row = sub.iloc[0]
        sector = row.get("Sector", "Unknown")
        val = row.get(metric)
        name = row.get("Name", t)

        st.subheader(f"{t} — {name}  (Sector: {sector})")
        st.plotly_chart(
            boxplot_sector_with_point(df, sector=sector, column=metric, point_value=val, label=t),
            use_container_width=True
        )
        hist = get_price_history_close(t)
        st.plotly_chart(price_timeseries(hist, title=f"Precio histórico de {t}"), use_container_width=True)

