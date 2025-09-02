import streamlit as st
from core.reporting import build_universe, score_and_rank
from ui.charts import bar_candidates_by_sector


def render():
    st.header("Dashboard")
    df = build_universe(10)
    df = score_and_rank(df)
    st.metric("Tickers analizados", len(df))
    st.plotly_chart(bar_candidates_by_sector(df), use_container_width=True)
