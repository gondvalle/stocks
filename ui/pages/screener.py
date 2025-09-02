import streamlit as st
import pandas as pd
from core.reporting import get_or_make_daily_scored

def render():
    st.header("Screener")
    with st.expander("Filtros", expanded=True):
        c1, c2, c3 = st.columns(3)
        min_score = c1.slider("Score mínimo", 0.0, 1.0, 0.5, 0.05)
        sector_sel = c2.text_input("Filtrar Sector (contiene)", "")
        contains = c3.text_input("Ticker / Nombre contiene", "")

    df = get_or_make_daily_scored(force_refresh=False)

    if sector_sel:
        df = df[df["Sector"].fillna("").str.contains(sector_sel, case=False, na=False)]
    if contains:
        m = df["Ticker"].str.contains(contains, case=False, na=False) | df["Name"].fillna("").str.contains(contains, case=False, na=False)
        df = df[m]
    df = df[df["score"] >= min_score].copy()

    st.caption(f"{len(df)} resultados")
    st.dataframe(df.sort_values("score", ascending=False), use_container_width=True)

    tt = st.selectbox("Ir al detalle de:", df["Ticker"].tolist() if not df.empty else [])
    if tt and st.button("Abrir detalle"):
        st.query_params["ticker"] = tt
