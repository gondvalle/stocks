# /sp500_screener/ui/pages/sectors.py
import streamlit as st
from core.reporting import build_universe, score_and_rank
from ui.charts import boxplot_by_sector

def render():
    st.header("Comparativas por Sector")
    with st.expander("Parámetros", expanded=True):
        c1, c2 = st.columns(2)
        max_names = c1.slider("Nº máximo", 50, 500, 200, step=50)
        metric = c2.selectbox("Métrica", ["pe_ttm","ev_ebitda_ttm","p_fcf","p_b","de_ratio","current_ratio","interest_coverage","fcf_margin","cfo_ni_ratio","roic","wacc_proxy","rev_cagr","eps_cagr"])

    with st.spinner("Cargando datos..."):
        df = score_and_rank(build_universe(max_names=max_names))

    st.plotly_chart(boxplot_by_sector(df, metric, title=f"{metric} por sector"), use_container_width=True)
