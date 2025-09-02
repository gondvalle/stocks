# /sp500_screener/ui/components.py
from __future__ import annotations
import streamlit as st

def badge(label: str, state: str) -> None:
    colors = {"ok": "green", "warn": "orange", "bad": "red"}
    color = colors.get(state, "gray")
    st.markdown(f"<span style='color:{color}'>●</span> {label}", unsafe_allow_html=True)

def show_check_line(label: str, value, condition: bool, fmt="{:.2f}"):
    icon = "✅" if condition else "❌"
    v = "n/d" if value is None else (fmt.format(value) if isinstance(value, (int, float)) else str(value))
    st.write(f"**{label}**: {v} {icon}")
