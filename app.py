# /sp500_screener/app.py
import streamlit as st
from ui.pages import dashboard, screener, stock_detail, sectors, settings

st.set_page_config(page_title="Screener S&P 500", layout="wide")

PAGES = {
    "Dashboard": dashboard,
    "Screener": screener,
    "Detalle de Acción": stock_detail,
    "Comparativas por Sector": sectors,
    "Ajustes": settings,
}

with st.sidebar:
    st.title("Navegación")
    page = st.radio("Ir a", list(PAGES.keys()))
    st.markdown("---")
    st.caption("Screener fundamental S&P 500 (Yahoo Finance / yfinance)")

# Permite navegar al detalle desde query params (?ticker=MSFT)
qp = st.query_params
if "ticker" in qp:
    st.session_state["selected_ticker"] = qp.get("ticker")

PAGES[page].render()
