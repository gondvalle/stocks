"""Componentes reutilizables para la UI."""
from __future__ import annotations

import streamlit as st


def badge(label: str, state: str) -> None:
    colors = {"ok": "green", "warn": "orange", "bad": "red"}
    color = colors.get(state, "gray")
    st.markdown(f"<span style='color:{color}'>●</span> {label}", unsafe_allow_html=True)
