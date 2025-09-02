import streamlit as st
import pandas as pd

from core.reporting import get_or_make_daily_scored, export_csv, save_daily_top_view
from ui.charts import bar_candidates_by_sector, scatter_two, heatmap_checks

def render():
    st.header("Dashboard")
    with st.expander("Parámetros", expanded=True):
        c1, c2 = st.columns([1,1])
        show_top = c1.slider("Mostrar TOP K", 5, 50, 10)
        force = c2.toggle("Recalcular hoy (forzar)", value=False)

    # Progreso: primera vez del día se verá 10/500, 20/500, etc.
    progress_txt = st.empty()
    progress_bar = st.progress(0, text="Preparando...")
    done_holder = {"last": 0}

    def _cb(done, total):
        pct = int(done/total*100)
        progress_bar.progress(pct, text=f"Descargando y calculando: {done}/{total}")
        progress_txt.info(f"Procesando tickers: {done}/{total}")
        done_holder["last"] = done

    df = get_or_make_daily_scored(force_refresh=force, workers=8, progress_cb=_cb)
    # Si cargó desde disco, no hubo callback
    if done_holder["last"] == 0:
        progress_bar.progress(100, text="Datos de hoy ya disponibles (500/500)")
        progress_txt.success("Cargados desde caché diaria.")

    st.metric("Tickers analizados (hoy)", len(df))
    st.plotly_chart(bar_candidates_by_sector(df), use_container_width=True)

    # TOP table
    top = df.head(show_top)
    st.subheader(f"Top {len(top)} por score compuesto")
    st.dataframe(top[["Ticker","Name","Sector","score","price","target_mean","forward_pe","pe_5y","pe_10y","ev_ebitda_ttm","p_fcf","roic","wacc_proxy"]], use_container_width=True)

    # Guardamos vista del día con el K visualizado
    saved_path = save_daily_top_view(df, show_top)
    st.caption(f"Vista guardada para hoy: `{saved_path}`")

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

