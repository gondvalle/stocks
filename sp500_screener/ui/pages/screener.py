import streamlit as st
from core.reporting import build_universe, score_and_rank


def render():
    st.header("Screener")
    df = score_and_rank(build_universe(10))
    st.dataframe(df)
