# /sp500_screener/ui/charts.py
from __future__ import annotations
import pandas as pd
import plotly.express as px
import numpy as np

def bar_candidates_by_sector(df: pd.DataFrame):
    d = df.groupby("Sector").size().reset_index(name="n").sort_values("n", ascending=False)
    fig = px.bar(d, x="Sector", y="n", title="Candidatos por sector")
    fig.update_layout(xaxis_title="", yaxis_title="Nº compañías", bargap=0.2)
    return fig

def boxplot_by_sector(df: pd.DataFrame, column: str, title: str | None = None):
    sub = df.dropna(subset=[column])
    if sub.empty:
        return px.scatter(title="Sin datos suficientes")
    fig = px.box(sub, x="Sector", y=column, points="outliers", title=title or f"{column} por sector")
    fig.update_layout(xaxis_title="", yaxis_title=column)
    return fig

def scatter_two(df: pd.DataFrame, x: str, y: str, color: str = "Sector", hover: list[str] | None = None, title: str | None = None):
    sub = df.dropna(subset=[x, y])
    if hover is None:
        hover = ["Ticker","Name"]
    fig = px.scatter(sub, x=x, y=y, color=color, hover_data=hover, title=title or f"{x} vs {y}")
    return fig

def heatmap_checks(df_checks: pd.DataFrame):
    """df_checks: DataFrame booleano de checks (filas=tickers, cols=checks)."""
    if df_checks.empty:
        return px.imshow(np.zeros((1,1)), title="Sin checks")
    # map booleans to 1/0 for color
    vals = df_checks.astype(int)
    fig = px.imshow(vals, aspect="auto", color_continuous_scale="Viridis", title="Mapa de checks (1=✅, 0=❌)")
    fig.update_xaxes(side="top")
    return fig
