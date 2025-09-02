# /sp500_screener/ui/pages/settings.py
import streamlit as st
import shutil
from core.config import CACHE_DIR
from core.reporting import get_or_make_daily_scored

def render():
    st.header("Ajustes")
    st.write("Las variables se leen de `.env` (RISK_FREE). Reinicia la app si cambias el `.env`.")

    if st.button("Invalidar caché en disco (yfinance)"):
        try:
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            st.success("Caché invalidada.")
        except Exception as e:
            st.error(f"No se pudo invalidar la caché: {e}")

    if st.button("Recalcular hoy (forzar)"):
        with st.spinner("Recalculando universo y guardando cache diaria..."):
            df = get_or_make_daily_scored(force_refresh=True)
        st.success(f"Recalculado {len(df)} tickers y guardado en caché diaria.")
