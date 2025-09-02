"""Funciones de gráficos."""
from __future__ import annotations

import plotly.express as px
import pandas as pd


def bar_candidates_by_sector(df: pd.DataFrame):
    return px.bar(df.groupby("Sector").size().reset_index(name="n"), x="Sector", y="n")


def boxplot_by_sector(df: pd.DataFrame, column: str):
    return px.box(df, x="Sector", y=column)


def scatter_pfcf_vs_fcfmargin(df: pd.DataFrame):
    return px.scatter(df, x="p_fcf", y="fcf_margin", color="Sector")


def heatmap_checks(df: pd.DataFrame):
    return px.imshow(df)
