# /sp500_screener/ui/pages/dashboard.py
import streamlit as st
import pandas as pd

from core.reporting import build_universe, score_and_rank, export_csv
from ui.charts import bar_candidates_by_sector, scatter_two, heatmap_checks

def render():
    st.header("Dashboard")
    with st.expander("Parámetros", expanded=True):
        c1, c2, c3 = st.columns([1,1,1])
        max_names = c1.slider("Nº máximo de compañías", 50, 500, 200, step=50)
        workers = c2.slider("Paralelismo (workers)", 1, 16, 8)
        show_top = c3.slider("Mostrar TOP K", 5, 50, 10)

    with st.spinner("Construyendo universo y calculando métricas..."):
        df = build_universe(max_names=max_names, max_workers=workers)
        df = score_and_rank(df)

    st.metric("Tickers analizados", len(df))
    st.plotly_chart(bar_candidates_by_sector(df), use_container_width=True)

    # TOP table
    top = df.head(show_top)
    st.subheader(f"Top {len(top)} por score compuesto")
    st.dataframe(top[["Ticker","Name","Sector","score","price","target_mean","forward_pe","pe_5y","pe_10y","ev_ebitda_ttm","p_fcf","roic","wacc_proxy"]], use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(scatter_two(df, "p_fcf", "fcf_margin", title="P/FCF vs FCF margin"), use_container_width=True)
    with c2:
        st.plotly_chart(scatter_two(df, "ev_ebitda_ttm", "roic", title="EV/EBITDA vs ROIC"), use_container_width=True)

    # Heatmap de checks del TOP
    st.subheader("Mapa de checks del Top")
    checks_df = pd.DataFrame(list(top["checks"])).astype(bool)
    checks_df.index = top["Ticker"]
    st.plotly_chart(heatmap_checks(checks_df), use_container_width=True)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Exportar CSV completo"):
            export_csv(df)
            st.success("Archivo guardado: sp500_top10_fundamental_screen.csv")
    with c2:
        ticker_go = st.selectbox("Ir al detalle de:", top["Ticker"].tolist())
        if st.button("Abrir detalle seleccionado"):
            st.query_params["ticker"] = ticker_go
            st.switch_page("app.py")  # queda en la misma app, usamos query param
