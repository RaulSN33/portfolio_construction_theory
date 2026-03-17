"""
attribution_dashboard.py
------------------------
Streamlit / Plotly UI layer for the Performance Attribution page.

Charts
------
  1. Scatter — stock vs market excess returns with static OLS regression line
  2. Rolling alpha & beta (dual subplot, shared x-axis)
  3. Cumulative return decomposition (actual / factor / idiosyncratic)
  4. Stacked area — % return attribution (factor vs idiosyncratic)
  5. Stacked area — % risk decomposition (R² systematic vs idiosyncratic)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.portfolio_construction.performance_attribution import AttributionResults

# ── Colour palette ────────────────────────────────────────────────────────────

ATTR = {
    "actual": "#666666",   # medium grey — actual return line (visible on light & dark)
    "factor": "#4e8bc4",   # steelblue   — factor / systematic / beta
    "idio":   "#ff9f40",   # orange      — idiosyncratic / alpha
    "zero":   "#888888",   # muted grey  — reference lines at 0 or 1
}

_FONT = "IBM Plex Mono"

# Base layout shared across all charts
_BASE = dict(
    font=dict(family=_FONT),
    legend=dict(
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=60, r=40, t=55, b=60),
    height=430,
    hovermode="x unified",
)

_AXIS_DEFAULTS = dict(title_font=dict(size=12))


def _axis(title: str, fmt: str | None = None, **extra) -> dict:
    d = dict(title=title, **_AXIS_DEFAULTS)
    if fmt:
        d["tickformat"] = fmt
    d.update(extra)
    return d


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_attribution_sidebar() -> dict:
    """
    Draw the attribution sidebar controls.

    Returns
    -------
    dict with keys: stock, start_date, end_date, window_size, run
    """
    with st.sidebar:
        st.markdown("## ⚙️ Attribution Setup")
        st.markdown("---")

        stock = st.text_input("Stock Ticker", value="QQQ", key="attr_stock").strip().upper()

        st.markdown("---")
        st.markdown("**Date Range**")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start", value=pd.Timestamp("2015-01-01"), key="attr_start"
            )
        with col2:
            end_date = st.date_input(
                "End", value=pd.Timestamp("today"), key="attr_end"
            )

        st.markdown("---")
        window_size = st.slider(
            "Rolling Window (months)",
            min_value=12,
            max_value=60,
            value=24,
            step=1,
            key="attr_window",
            help="Number of months used for each rolling OLS regression.",
        )

        st.markdown("---")
        run = st.button("▶  Run Attribution", use_container_width=True, key="attr_run")

    return dict(
        stock=stock,
        start_date=start_date,
        end_date=end_date,
        window_size=window_size,
        run=run,
    )


# ── Page header ───────────────────────────────────────────────────────────────

def render_attribution_header(stock: str, window_size: int) -> None:
    st.markdown(f"# Performance Attribution — {stock}")
    st.markdown(
        f"*Rolling {window_size}-month CAPM regression · "
        "Return & risk decomposition vs S&P 500*"
    )
    st.markdown("---")


# ── Static OLS metric cards ───────────────────────────────────────────────────

def render_attribution_metrics(res: AttributionResults) -> None:
    """Four metric cards showing full-sample OLS results."""
    st.markdown("### Full-Sample CAPM (Static OLS)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Alpha (monthly)", f"{res.static_alpha:.4%}")
    with c2:
        st.metric("Beta", f"{res.static_beta:.3f}")
    with c3:
        st.metric("R²", f"{res.static_r2:.3f}")
    with c4:
        st.metric("Observations", len(res.excess_returns))
    st.markdown("---")


# ── Chart 1: Scatter + static OLS line ───────────────────────────────────────

def build_scatter_chart(res: AttributionResults) -> go.Figure:
    """
    Scatter of stock vs market excess returns with the static OLS regression
    line overlaid. Zero-axis reference lines added.
    """
    market    = res.excess_returns[res.excess_returns.columns[-1]]   # ^GSPC
    stock_ret = res.excess_returns[res.stock]

    # Regression line spanning the full x range
    x_min, x_max = float(market.min()), float(market.max())
    x_line = np.linspace(x_min, x_max, 200)
    y_line = res.static_alpha + res.static_beta * x_line

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=market,
        y=stock_ret,
        mode="markers",
        name="Monthly Returns",
        marker=dict(
            color=ATTR["factor"],
            size=7,
            opacity=0.75,
            line=dict(color="#ffffff", width=0.4),
        ),
        hovertemplate="Market: %{x:.2%}<br>Stock: %{y:.2%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name=f"OLS fit  α={res.static_alpha:.3%}  β={res.static_beta:.2f}",
        line=dict(color=ATTR["idio"], width=2),
        hoverinfo="skip",
    ))

    fig.add_hline(y=0, line=dict(color=ATTR["zero"], width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=ATTR["zero"], width=1, dash="dot"))

    fig.update_layout(
        **_BASE,
        title=dict(
            text=f"{res.stock} vs S&P 500 — Monthly Excess Returns",
            font=dict(size=14),
        ),
        xaxis=_axis("S&P 500 Excess Return", ".1%"),
        yaxis=_axis(f"{res.stock} Excess Return", ".1%"),
        # hovermode="closest",
    )
    return fig


# ── Chart 2: Rolling alpha & beta ────────────────────────────────────────────

def build_rolling_params_chart(res: AttributionResults) -> go.Figure:
    """
    Two-row subplot: beta (top) with reference line at 1,
    alpha (bottom) with reference line at 0.
    """
    rr = res.rolling

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("Beta", "Alpha (monthly)"),
    )

    fig.add_trace(go.Scatter(
        x=rr.index,
        y=rr["beta"],
        mode="lines",
        name="Beta",
        line=dict(color=ATTR["factor"], width=1.8),
        hovertemplate="%{x|%Y-%m}<br>β = %{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(
        y=1,
        line=dict(color=ATTR["zero"], width=1, dash="dot"),
        row=1, col=1,
    )

    fig.add_trace(go.Scatter(
        x=rr.index,
        y=rr["alpha"],
        mode="lines",
        name="Alpha",
        line=dict(color=ATTR["idio"], width=1.8),
        hovertemplate="%{x|%Y-%m}<br>α = %{y:.3%}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(
        y=0,
        line=dict(color=ATTR["zero"], width=1, dash="dot"),
        row=2, col=1,
    )

    fig.update_layout(
        **{**_BASE, "height": 520},
        title=dict(
            text=f"{res.stock} — Rolling CAPM Parameters ({res.window_size}m window)",
            font=dict(size=14),
        ),
        showlegend=True,
    )
    fig.update_yaxes(title_text="Beta",  row=1, col=1)
    fig.update_yaxes(title_text="Alpha", tickformat=".2%", row=2, col=1)

    # Style the auto-generated subplot titles
    for ann in fig.layout.annotations:
        ann.font.family = _FONT
        ann.font.size   = 12

    return fig


# ── Chart 3: Cumulative return decomposition ──────────────────────────────────

def build_cumulative_decomp_chart(res: AttributionResults) -> go.Figure:
    """
    Three lines showing how actual, factor, and idiosyncratic returns
    have cumulatively accumulated over the rolling window period.
    """
    cs = res.cumsum_decomp

    fig = go.Figure()

    for col, color, label in [
        ("actual_return", ATTR["actual"], "Actual Return"),
        ("factor_return", ATTR["factor"], "Factor Return"),
        ("idio_return",   ATTR["idio"],   "Idiosyncratic Return"),
    ]:
        fig.add_trace(go.Scatter(
            x=cs.index,
            y=cs[col],
            mode="lines",
            name=label,
            line=dict(color=color, width=1.8),
            hovertemplate=f"%{{x|%Y-%m}}<br>{label}: %{{y:.2%}}<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=ATTR["zero"], width=1, dash="dot"))

    fig.update_layout(
        **_BASE,
        title=dict(
            text=(
                f"{res.stock} — Cumulative Return Decomposition "
                f"(Rolling {res.window_size}m CAPM)"
            ),
            font=dict(size=14),
        ),
        xaxis=_axis(""),
        yaxis=_axis("Cumulative Return", ".0%"),
    )
    return fig


# ── Chart 4: Return attribution stacked area ──────────────────────────────────

def build_return_attribution_chart(res: AttributionResults) -> go.Figure:
    """
    Stacked area showing the % share of total absolute return
    explained by the factor vs idiosyncratic component each month.
    """
    rd        = res.return_decomp
    abs_total = rd[["factor_return", "idio_return"]].abs().sum(axis=1)
    pct_factor = (rd["factor_return"].abs() / abs_total).fillna(0.5)
    pct_idio   = (rd["idio_return"].abs()   / abs_total).fillna(0.5)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rd.index,
        y=pct_factor,
        mode="lines",
        name="Factor (Market)",
        stackgroup="one",
        line=dict(width=0),
        fillcolor="rgba(78,139,196,0.72)",
        hovertemplate="%{x|%Y-%m}<br>Factor: %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=rd.index,
        y=pct_idio,
        mode="lines",
        name="Idiosyncratic",
        stackgroup="one",
        line=dict(width=0),
        fillcolor="rgba(255,159,64,0.72)",
        hovertemplate="%{x|%Y-%m}<br>Idiosyncratic: %{y:.1%}<extra></extra>",
    ))

    fig.update_layout(
        **_BASE,
        title=dict(
            text=(
                f"{res.stock} — Return Attribution: "
                "Factor vs Idiosyncratic (% of abs. total)"
            ),
            font=dict(size=14),
        ),
        xaxis=_axis(""),
        yaxis=_axis("Share of Total Return", ".0%", range=[0, 1]),
    )
    return fig


# ── Chart 5: Risk decomposition stacked area ──────────────────────────────────

def build_risk_decomposition_chart(res: AttributionResults) -> go.Figure:
    """
    Stacked area showing systematic (R²) vs idiosyncratic (1−R²) share
    of total return variance in each rolling window.
    """
    rr = res.rolling

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rr.index,
        y=rr["r_squared"],
        mode="lines",
        name="Systematic Risk (R²)",
        stackgroup="one",
        line=dict(width=0),
        fillcolor="rgba(78,139,196,0.72)",
        hovertemplate="%{x|%Y-%m}<br>Systematic: %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=rr.index,
        y=1 - rr["r_squared"],
        mode="lines",
        name="Idiosyncratic Risk (1 − R²)",
        stackgroup="one",
        line=dict(width=0),
        fillcolor="rgba(255,159,64,0.72)",
        hovertemplate="%{x|%Y-%m}<br>Idiosyncratic: %{y:.1%}<extra></extra>",
    ))

    fig.update_layout(
        **_BASE,
        title=dict(
            text=(
                f"{res.stock} — Risk Decomposition: "
                f"Systematic vs Idiosyncratic ({res.window_size}m rolling)"
            ),
            font=dict(size=14),
        ),
        xaxis=_axis(""),
        yaxis=_axis("Share of Total Variance", ".0%", range=[0, 1]),
    )
    return fig


# ── Render all charts in sequence ─────────────────────────────────────────────

def render_attribution_charts(res: AttributionResults) -> None:
    st.markdown("### Returns vs Market (Static CAPM)")
    st.plotly_chart(build_scatter_chart(res), use_container_width=True)
    st.markdown("---")

    st.markdown("### Rolling Alpha & Beta")
    st.plotly_chart(build_rolling_params_chart(res), use_container_width=True)
    st.markdown("---")

    st.markdown("### Cumulative Return Decomposition")
    st.plotly_chart(build_cumulative_decomp_chart(res), use_container_width=True)
    st.markdown("---")

    st.markdown("### Return Attribution")
    st.plotly_chart(build_return_attribution_chart(res), use_container_width=True)
    st.markdown("---")

    st.markdown("### Risk Decomposition")
    st.plotly_chart(build_risk_decomposition_chart(res), use_container_width=True)
    st.caption(
        f"Data: Yahoo Finance · FRED GS10 · Interval: Monthly · "
        f"Rolling window: {res.window_size} months"
    )
