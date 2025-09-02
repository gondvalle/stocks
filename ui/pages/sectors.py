import streamlit as st
from core.reporting import build_universe
from core.sectors import sector_medians


def render():
    st.header("Sectores")
    df = build_universe(10)
    meds = sector_medians(df)
    st.write(meds)
