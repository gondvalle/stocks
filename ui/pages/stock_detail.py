import streamlit as st
from core.fetch import get_price_and_target


def render(ticker: str | None = None):
    st.header("Detalle de Acción")
    if ticker is None:
        st.info("Seleccione un ticker desde el screener.")
        return
    price, target = get_price_and_target(ticker)
    st.write(f"Ticker: {ticker} - Precio: {price} - Target: {target}")
