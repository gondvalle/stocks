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

st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a", list(PAGES.keys()))

PAGES[page].render()
