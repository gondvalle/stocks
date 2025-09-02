# ui/charts.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List

# =========================
# Gráficos existentes (dashboard / sectores)
# =========================

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

def scatter_two(df: pd.DataFrame, x: str, y: str, color: str = "Sector", hover: List[str] | None = None, title: str | None = None):
    sub = df.dropna(subset=[x, y])
    if hover is None:
        hover = ["Ticker","Name"]
    fig = px.scatter(sub, x=x, y=y, color=color, hover_data=hover, title=title or f"{x} vs {y}")
    return fig

def heatmap_checks(df_checks: pd.DataFrame):
    """df_checks: DataFrame booleano de checks (filas=tickers, cols=checks)."""
    if df_checks.empty:
        return px.imshow(np.zeros((1,1)), title="Sin checks")
    vals = df_checks.astype(int)
    fig = px.imshow(vals, aspect="auto", color_continuous_scale="Viridis", title="Mapa de checks (1=✅, 0=❌)")
    fig.update_xaxes(side="top")
    return fig

def boxplot_sector_with_point(df_scored: pd.DataFrame, sector: str, column: str, point_value: float, label: str) -> go.Figure:
    """Boxplot de una métrica por sector con un punto resaltando la compañía."""
    sub = df_scored[df_scored["Sector"] == sector].copy()
    sub = sub.dropna(subset=[column])
    if sub.empty or not np.isfinite(point_value):
        return px.scatter(title=f"Sin datos suficientes para {column}")
    fig = px.box(sub, x="Sector", y=column, points="outliers", title=f"{column} — {sector}")
    fig.add_trace(go.Scatter(
        x=[sector], y=[point_value], mode="markers+text", text=[label],
        textposition="top center", marker=dict(size=12, symbol="circle-open")
    ))
    fig.update_layout(xaxis_title="", yaxis_title=column)
    return fig

def price_timeseries(close_series: pd.Series, title: Optional[str] = None) -> go.Figure:
    if close_series is None or close_series.empty:
        return px.scatter(title="Sin histórico de precio")
    df = close_series.reset_index()
    df.columns = ["Date", "Close"]
    fig = px.line(df, x="Date", y="Close", title=title or "Precio histórico (cierre)")
    return fig

# =========================
# Nuevos gráficos empresa vs sector (lo que pediste)
# =========================

def bar_forwardpe_vs_hist(row: dict, sector_meds: dict) -> go.Figure:
    """
    Muestra Forward P/E frente a P/E históricos (5y/10y) y línea con mediana sector (P/E TTM).
    """
    fpe = row.get("forward_pe", np.nan)
    pe5 = row.get("pe_5y", np.nan)
    pe10 = row.get("pe_10y", np.nan)
    sect = row.get("Sector", "Unknown")
    sect_pe_ttm_median = None
    if sector_meds and sect in sector_meds:
        sect_pe_ttm_median = sector_meds[sect].get("pe_ttm_median")

    dfb = pd.DataFrame({
        "Métrica": ["Forward P/E", "PE 5y", "PE 10y"],
        "Valor": [fpe, pe5, pe10]
    })
    fig = px.bar(dfb, x="Métrica", y="Valor", title="Forward P/E vs P/E histórico (5/10 años)")
    fig.update_layout(yaxis_title="P/E")

    if sect_pe_ttm_median is not None and sect_pe_ttm_median == sect_pe_ttm_median:
        fig.add_hline(
            y=sect_pe_ttm_median, line_dash="dash",
            annotation_text=f"Mediana sector P/E TTM: {sect_pe_ttm_median:.2f}",
            annotation_position="top left"
        )
    return fig

def boxplot_sector_metric(df_scored: pd.DataFrame, sector: str, column: str, point_value: float, label: str, title: str, y_title: str) -> go.Figure:
    """
    Helper genérico para dibujar un boxplot de 'column' por sector y resaltar el valor de la compañía.
    """
    sub = df_scored[df_scored["Sector"] == sector].copy()
    sub = sub.dropna(subset=[column])
    if sub.empty or not np.isfinite(point_value):
        return px.scatter(title=f"Sin datos suficientes para {title}")
    fig = px.box(sub, x="Sector", y=column, points="outliers", title=title)
    fig.add_trace(go.Scatter(
        x=[sector], y=[point_value], mode="markers+text", text=[label],
        textposition="top center", marker=dict(size=12, symbol="circle-open")
    ))
    fig.update_layout(xaxis_title="", yaxis_title=y_title)
    return fig

def scatter_cfo_ni_vs_de(df_scored: pd.DataFrame, sector: str, ticker: str) -> go.Figure:
    """
    X = D/E (menor es mejor), Y = CFO/NI (mayor es mejor).
    Pinta todo el sector y resalta la compañía.
    """
    sub = df_scored[df_scored["Sector"] == sector].copy()
    if sub.empty:
        return px.scatter(title="Sin datos del sector para CFO/NI vs D/E")

    sub = sub[["Ticker","Name","de_ratio","cfo_ni_ratio"]].dropna()
    if sub.empty:
        return px.scatter(title="Sin datos suficientes para CFO/NI vs D/E")

    fig = px.scatter(sub, x="de_ratio", y="cfo_ni_ratio", hover_data=["Ticker","Name"],
                     title="Cashflow vs Deuda (CFO/NI vs D/E)")
    fig.update_layout(xaxis_title="D/E (menor es mejor)", yaxis_title="CFO/NI (mayor es mejor)")

    me = sub[sub["Ticker"] == ticker]
    if not me.empty:
        fig.add_trace(go.Scatter(
            x=me["de_ratio"], y=me["cfo_ni_ratio"], mode="markers+text",
            text=[ticker], textposition="top center",
            marker=dict(size=14, symbol="star")
        ))
    return fig

def scatter_fcf_margin_vs_de(df_scored: pd.DataFrame, sector: str, ticker: str) -> go.Figure:
    """
    X = D/E (menor es mejor), Y = FCF margin (mayor es mejor).
    """
    sub = df_scored[df_scored["Sector"] == sector].copy()
    if sub.empty:
        return px.scatter(title="Sin datos del sector para FCF margin vs D/E")

    sub = sub[["Ticker","Name","de_ratio","fcf_margin"]].dropna()
    if sub.empty:
        return px.scatter(title="Sin datos suficientes para FCF margin vs D/E")

    fig = px.scatter(sub, x="de_ratio", y="fcf_margin", hover_data=["Ticker","Name"],
                     title="Calidad de caja vs deuda (FCF margin vs D/E)")
    fig.update_layout(xaxis_title="D/E (menor es mejor)", yaxis_title="FCF margin (mayor es mejor)")

    me = sub[sub["Ticker"] == ticker]
    if not me.empty:
        fig.add_trace(go.Scatter(
            x=me["de_ratio"], y=me["fcf_margin"], mode="markers+text",
            text=[ticker], textposition="top center",
            marker=dict(size=14, symbol="star")
        ))
    return fig
