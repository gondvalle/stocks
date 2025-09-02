import streamlit as st


def render():
    st.header("Ajustes")
    if st.button("Recargar datos desde Yahoo"):
        st.success("Caché invalidado (demo)")
