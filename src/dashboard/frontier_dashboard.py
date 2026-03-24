"""
frontier_dashboard.py
------------
Streamlit / Plotly UI layer.
All page configuration, CSS injection, sidebar rendering, metric cards,
data tables, and chart building live here.
"""
from __future__ import annotations

from datetime import timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.portfolio_construction.frontier import FrontierResults, sharpe
from src.portfolio_construction.optimizations import ConstrainedResults

# ── Colour palette ───────────────────────────────────────────────────────────

COLORS = {
    "frontier":             "#888888",
    "frontier_constrained": "#4e8bc4",
    "cml":                  "#ff6b6b",
    "assets":               "#7ec8e3",
    "gmv":                  "#f0e68c",
    "gmv_constrained":      "#ffa94d",
    "tangency":             "#90ee90",
    "tangency_constrained": "#69d2e7",
    "rf":                   "#ff6b6b",
    "ew":                   "#c9a0ff",
}
_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    border-radius: 6px;
    padding: 12px 16px;
  }
  div[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
  }

  /* Dataframe */
  .stDataFrame { border-radius: 6px; }
  .stDataFrame th {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }
</style>
"""


# ── Page setup ───────────────────────────────────────────────────────────────

def setup_page() -> None:
    st.set_page_config(
        page_title="Efficient Frontier",
        page_icon="📈",
        layout="wide",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Draw the sidebar controls and return a dict of user inputs:
        tickers_raw, start_date, end_date, run

    RF rate and analysis date range are handled in the main area via
    render_analysis_controls() so they don't trigger a data re-download.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Data Source")
        st.markdown("---")

        tickers_raw = st.text_area(
            "Tickers (one per line or comma-separated)",
            value="F\nPFE\nCVX\nWMT\nSBUX\nAMZN\nDIS\nCOST\nMMM",
            height=200,
        )

        st.markdown("---")
        st.markdown("**Download Date Range**")
        start_date = st.date_input("Start", value=pd.Timestamp("2015-01-01"))

        st.markdown("---")
        run = st.button("Run Analysis!", use_container_width=True)

    return dict(
        tickers_raw=tickers_raw,
        start_date=start_date,
        run=run,
    )


# ── Header ───────────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown("# Efficient Frontier")
    st.markdown("*Monthly historical risk-return tradeoff with Capital Market Line*")
    st.markdown("---")


# ── Analysis controls (main area) ────────────────────────────────────────────

def render_analysis_controls(
    prices_index: pd.DatetimeIndex,
    default_rf: float = 0.035,
) -> dict:
    """
    Render the interactive analysis controls in the main content area.
    These do NOT trigger a data re-download — only a frontier recomputation.

    Returns
    -------
    dict with keys: end, rf_rate
    """
    st.markdown("### Analysis Parameters")
    st.caption(
        "Set the backtesting start date and risk-free rate. "
        "Stats are computed on all history up to that date; "
        "the buy-and-hold period runs from that date to today."
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        analysis_end = st.date_input(
            "Backtesting Start",
            value=prices_index.max().date() - timedelta(days = 366),
            min_value=prices_index.min().date(),
            max_value=prices_index.max().date(),
            key="analysis_end",
        )
        # st.markdown(type(prices_index.max().date()))
        # st.markdown()
    with col2:
        rf_rate = st.number_input(
            "Annual Risk-Free Rate",
            min_value=0.0,
            max_value=0.1,
            value=default_rf,
            step=0.0001,
            format="%.4f",
            key="rf_rate",
        )/252

    st.markdown("---")
    return dict(end=analysis_end, rf_rate=rf_rate)


# ── Metric cards ─────────────────────────────────────────────────────────────

def render_metrics(
    valid_stocks: list[str],
    returns: pd.DataFrame,
    results: FrontierResults,
    rf_rate: float,
) -> None:
    st.markdown("### Portfolio Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Tickers", len(valid_stocks))
    with c2:
        st.metric("Days", len(returns))
    with c3:
        st.metric("GMV Return", f"{results.gmv_ret:.2%}")
    with c4:
        st.metric("GMV Volatility", f"{results.gmv_std:.2%}")
    with c5:
        st.metric("Tangency Return", f"{results.tan_ret:.2%}")
    with c6:
        st.metric(
            "Tangency Sharpe",
            f"{sharpe(results.tan_ret, results.tan_std, rf_rate):.2f}",
        )
    st.markdown("---")


# ── Analytics & weights tables ───────────────────────────────────────────────

def render_tables(
    valid_stocks: list[str],
    results: FrontierResults,
    rf_rate: float,
    constrained: ConstrainedResults | None = None,
) -> None:
    portfolios = {
        "GMV (unconstrained)":         (results.w_gmv, results.gmv_ret, results.gmv_std),
        "Max Sharpe (unconstrained)":  (results.w_tan, results.tan_ret, results.tan_std),
        "Equal Weight":                (results.w_ew,  results.ew_ret,  results.ew_std),
    }
    weights_dict = {
        "EW":              results.w_ew,
        "GMV":             results.w_gmv,
        "Max Sharpe":      results.w_tan,
    }

    if constrained is not None:
        portfolios["GMV (constrained)"]        = (constrained.w_gmv, constrained.gmv_ret, constrained.gmv_std)
        portfolios["Max Sharpe (constrained)"] = (constrained.w_tan, constrained.tan_ret, constrained.tan_std)
        weights_dict["C-GMV"]        = constrained.w_gmv
        weights_dict["C-Max Sharpe"] = constrained.w_tan

    rows = []
    for name, (_, ret, std) in portfolios.items():
        rows.append({
            "Portfolio":       name,
            "Mean Daily Return":  f"{ret:.4%}",
            "Daily Std Dev": f"{std:.4%}",
            "Daily Sharpe Ratio":    f"{sharpe(ret, std, rf_rate):.3f}",
        })

    c11, c21 = st.columns(2)
    with c11:
        st.markdown("### Portfolio Analytics")
        st.dataframe(
            pd.DataFrame(rows).set_index("Portfolio"),
            use_container_width=True,
        )
    with c21:
        st.markdown("### Portfolio Weights")
        st.dataframe(
            pd.DataFrame(weights_dict, index=valid_stocks)*100,
            use_container_width=True,
            column_config={
                "EW": st.column_config.NumberColumn(format="%.2f%%"),
                "GMV": st.column_config.NumberColumn(format="%.2f%%"),
                "Max Sharpe": st.column_config.NumberColumn(format="%.2f%%"),
                "C-GMV": st.column_config.NumberColumn(format="%.2f%%"),
                "C-Max Sharpe": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )
    st.markdown("---")


# ── Plotly chart ─────────────────────────────────────────────────────────────

def build_frontier_chart(
    results: FrontierResults,
    miu: pd.Series,
    std_dev: pd.Series,
    valid_stocks: list[str],
    rf_rate: float,
    constrained: ConstrainedResults | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Efficient frontier (unconstrained)
    fig.add_trace(go.Scatter(
        x=results.frontier_vols, y=results.frontier_rets,
        mode="lines",
        name="Efficient Frontier (unconstrained)",
        line=dict(color=COLORS["frontier"], width=2.5),
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # Constrained frontier
    if constrained is not None:
        fig.add_trace(go.Scatter(
            x=constrained.frontier_vols, y=constrained.frontier_rets,
            mode="lines",
            name="Efficient Frontier (constrained)",
            line=dict(color=COLORS["frontier_constrained"], width=2.5, dash="dot"),
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        ))

    # CML
    fig.add_trace(go.Scatter(
        x=results.cml_vols, y=results.cml_rets,
        mode="lines",
        name="Capital Market Line",
        line=dict(color=COLORS["cml"], width=2, dash="dash"),
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # Individual assets
    fig.add_trace(go.Scatter(
        x=std_dev.values, y=miu.values,
        mode="markers+text",
        name="Assets",
        marker=dict(color=COLORS["assets"], size=9, symbol="circle",
                    line=dict(color="#ffffff", width=0.5)),
        text=valid_stocks,
        textposition="top right",
        textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["assets"]),
        hovertemplate="<b>%{text}</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # Equal weight
    fig.add_trace(go.Scatter(
        x=[results.ew_std], y=[results.ew_ret],
        mode="markers+text",
        name="Equal Weight",
        marker=dict(color=COLORS["ew"], size=16, symbol="diamond",
                    line=dict(color="#ffffff", width=1)),
        text=["1/N"],
        textposition="top right",
        textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["ew"]),
        hovertemplate="<b>Equal Weight</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # GMV (unconstrained)
    fig.add_trace(go.Scatter(
        x=[results.gmv_std], y=[results.gmv_ret],
        mode="markers+text",
        name="GMV (unconstrained)",
        marker=dict(color=COLORS["gmv"], size=18, symbol="star",
                    line=dict(color="#ffffff", width=1)),
        text=["GMV"],
        textposition="top right",
        textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["gmv"]),
        hovertemplate="<b>GMV (unconstrained)</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # Tangency / Max Sharpe (unconstrained)
    fig.add_trace(go.Scatter(
        x=[results.tan_std], y=[results.tan_ret],
        mode="markers+text",
        name="Max Sharpe (unconstrained)",
        marker=dict(color=COLORS["tangency"], size=18, symbol="star",
                    line=dict(color="#ffffff", width=1)),
        text=["Tangency"],
        textposition="top right",
        textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["tangency"]),
        hovertemplate="<b>Tangency</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
    ))

    # Constrained GMV and Max Sharpe
    if constrained is not None:
        fig.add_trace(go.Scatter(
            x=[constrained.gmv_std], y=[constrained.gmv_ret],
            mode="markers+text",
            name="GMV (constrained)",
            marker=dict(color=COLORS["gmv_constrained"], size=18, symbol="star",
                        line=dict(color="#ffffff", width=1)),
            text=["C-GMV"],
            textposition="top right",
            textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["gmv_constrained"]),
            hovertemplate="<b>GMV (constrained)</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[constrained.tan_std], y=[constrained.tan_ret],
            mode="markers+text",
            name="Max Sharpe (constrained)",
            marker=dict(color=COLORS["tangency_constrained"], size=18, symbol="star",
                        line=dict(color="#ffffff", width=1)),
            text=["C-Tangency"],
            textposition="top right",
            textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["tangency_constrained"]),
            hovertemplate="<b>Tangency (constrained)</b><br>Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        ))

    # Risk-free rate
    fig.add_trace(go.Scatter(
        x=[0], y=[rf_rate],
        mode="markers+text",
        name="Risk-Free Rate",
        marker=dict(color=COLORS["rf"], size=12, symbol="circle",
                    line=dict(color="#ffffff", width=1)),
        text=["Rf"],
        textposition="top right",
        textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["rf"]),
        hovertemplate=f"<b>Risk-Free Rate</b><br>{rf_rate:.4%}<extra></extra>",
    ))

    fig.update_layout(
        font=dict(family="IBM Plex Mono"),
        xaxis=dict(
            title="Daily Volatility (Std Dev)",
            tickformat=".1%",
            title_font=dict(size=12),
        ),
        yaxis=dict(
            title="Daily Return",
            tickformat=".2%",
            title_font=dict(size=12),
        ),
        legend=dict(
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=40, t=40, b=60),
        height=580,
        hovermode="closest",
    )
    return fig


def render_chart(
    results: FrontierResults,
    miu: pd.Series,
    std_dev: pd.Series,
    valid_stocks: list[str],
    rf_rate: float,
    constrained: ConstrainedResults | None = None,
) -> None:
    st.markdown("### Efficient Frontier & Capital Market Line")
    fig = build_frontier_chart(results, miu, std_dev, valid_stocks, rf_rate, constrained)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Data: Yahoo Finance · Interval: Daily · RF Rate: {rf_rate:.4%}/Day"
    )


# ── Backtest chart ────────────────────────────────────────────────────────────

_BACKTEST_COLORS = {
    "GMV (Unconstrained)":        COLORS["gmv"],
    "Max Sharpe (Unconstrained)": COLORS["tangency"],
    "Equal Weight":               COLORS["ew"],
    "GMV (Constrained)":          COLORS["gmv_constrained"],
    "Max Sharpe (Constrained)":   COLORS["tangency_constrained"],
}


def render_backtest_section(
    cumulative: dict,
    metrics_df: pd.DataFrame,
    backtest_start: str,
) -> None:
    """
    Render the buy-and-hold backtest cumulative performance chart
    and the concatenated performance metrics table.

    Parameters
    ----------
    cumulative     : dict mapping portfolio name → price_simulation Series
    metrics_df     : DataFrame with one row per portfolio (from summary_stats)
    backtest_start : ISO date string for the caption
    """
    st.markdown("---")
    st.markdown("### Buy-and-Hold Backtest")
    st.caption(
        f"Portfolios held from {backtest_start} to today · "
        "initial capital = 1 · Daily returns"
    )

    # ── Cumulative performance chart ─────────────────────────────────────────
    fig = go.Figure()
    for name, series in cumulative.items():
        color = _BACKTEST_COLORS.get(name, "#aaaaaa")
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>Value: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        font=dict(family="IBM Plex Mono"),
        xaxis=dict(title="Date", title_font=dict(size=12)),
        yaxis=dict(title="Portfolio Value (start = 1)", title_font=dict(size=12)),
        legend=dict(borderwidth=1, font=dict(size=11)),
        margin=dict(l=60, r=40, t=40, b=60),
        height=460,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Performance metrics table ─────────────────────────────────────────────
    st.markdown("### Backtested Performance Metrics")
    fmt_pct  = lambda x: f"{x:.2%}"
    fmt_2f   = lambda x: f"{x:.2f}"
    fmt_cols = {
        "Annualized Return": fmt_pct,
        "Annualized Vol": fmt_pct,
        "Sharpe Ratio": fmt_2f,
        "Max Drawdown": fmt_pct,
        "Skewness": fmt_2f,
        "Kurtosis": fmt_2f,
        "Cornish-Fisher VaR (5%)": fmt_pct,
        "Historic CVaR (5%)": fmt_pct,
    }
    styled = metrics_df.copy()
    for col, fn in fmt_cols.items():
        if col in styled.columns:
            styled[col] = styled[col].map(fn)
    st.dataframe(styled, use_container_width=True)


# ── Fama-French 3-Factor Attribution ─────────────────────────────────────────

_ATTR_COLORS = {
    "actual": "#888888",
    "factor": "#4e8bc4",
    "idio":   "#ff9f40",
    "mkt":    "#4e8bc4",
    "smb":    "#F24C4B",
    "hml":    "#27ae60",
    "alpha":  "#8e44ad",
}


def _hex_rgba(hex_color: str, alpha: float = 0.75) -> str:
    """Convert a hex color string to an rgba(...) string for Plotly fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_attribution_controls(portfolio_names: list[str]) -> dict:
    """
    Render the portfolio selector, rolling window slider, and run button
    for the FF3 attribution section.

    Returns
    -------
    dict with keys: portfolio (str), window_size (int), run (bool)
    """
    st.markdown("---")
    st.markdown("### Fama-French 3-Factor Performance Attribution")
    st.caption(
        "Decompose portfolio returns into Market (Mkt-RF), Size (SMB), Value (HML), "
        "Alpha, and Idiosyncratic components using rolling OLS on a Vanguard-ETF-constructed "
        "factor model."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected = st.selectbox(
            "Select Portfolio",
            options=portfolio_names,
            key="attr_portfolio_select",
        )
    with col2:
        window_size = st.slider(
            "Rolling Window (trading days)",
            min_value=126,
            max_value=756,
            value=504,
            step=63,
            key="attr_window_size",
            help="1 year ≈ 252 days. Default is 2 years (504 days).",
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ Run Attribution", use_container_width=True, key="attr_run")

    return dict(portfolio=selected, window_size=window_size, run=run)


def build_cumulative_decomp_chart(
    portfolio_results: pd.DataFrame,
    portfolio_name: str,
    window_size: int,
) -> go.Figure:
    """
    Two-panel cumulative return decomposition chart.
    Top panel  : Actual return vs Factor return vs Idiosyncratic return.
    Bottom panel: Individual factor contributions (Market, SMB, HML, Alpha).
    """
    cumsum = portfolio_results[[
        "actual_return", "factor_return", "idio_return",
        "ret_mkt", "ret_smb", "ret_hml", "ret_alpha",
    ]].cumsum()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Actual vs Factor vs Idiosyncratic",
            "Factor Components  (Market · SMB · HML)",
        ),
    )

    # ── Top panel ─────────────────────────────────────────────────────────────
    top_traces = [
        ("actual_return", "Actual Return",        _ATTR_COLORS["actual"], "solid", 2.5),
        ("factor_return", "Total Factor Return",  _ATTR_COLORS["factor"], "solid", 2.0),
        ("idio_return",   "Idiosyncratic Return", _ATTR_COLORS["idio"],   "solid",  1.5),
    ]
    for col, label, color, dash, width in top_traces:
        fig.add_trace(go.Scatter(
            x=cumsum.index, y=cumsum[col],
            name=label, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2%}}<extra></extra>",
        ), row=1, col=1)

    fig.add_hline(y=0, line=dict(color="#666666", dash="dot", width=0.8), row=1, col=1)

    # ── Bottom panel ──────────────────────────────────────────────────────────
    bottom_traces = [
        ("ret_mkt",   "Market (Mkt-RF)", _ATTR_COLORS["mkt"],   "solid", 1.8),
        ("ret_smb",   "Size (SMB)",      _ATTR_COLORS["smb"],   "solid", 1.8),
        ("ret_hml",   "Value (HML)",     _ATTR_COLORS["hml"],   "solid", 1.8),
        # ("ret_alpha", "Alpha",           _ATTR_COLORS["alpha"], "solid",  1.5),
    ]
    for col, label, color, dash, width in bottom_traces:
        fig.add_trace(go.Scatter(
            x=cumsum.index, y=cumsum[col],
            name=label, mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2%}}<extra></extra>",
        ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color="#666666", dash="dot", width=0.8), row=2, col=1)

    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(
        font=dict(family="IBM Plex Mono"),
        title=dict(
            text=f"{portfolio_name}; Cumulative Return Decomposition (Rolling {window_size}d FF3)",
            font=dict(size=14),
        ),
        height=680,
        hovermode="x unified",
        legend=dict(borderwidth=1, font=dict(size=11)),
        margin=dict(l=60, r=40, t=80, b=60),
    )
    return fig


def build_return_attribution_chart(
    portfolio_results: pd.DataFrame,
    portfolio_name: str,
    window_size: int,
    smoothing_days: int = 5,
) -> go.Figure:
    """
    Stacked area chart showing the % share of each factor in the total
    absolute return (smoothed with a rolling mean to reduce noise).
    """
    abs_components = portfolio_results[
        ["ret_mkt", "ret_smb", "ret_hml", "ret_alpha", "idio_return"]
    ].abs()
    abs_total = abs_components.sum(axis=1)
    pct = (
        abs_components.div(abs_total, axis=0)
        .rolling(smoothing_days)
        .mean()
        .dropna()
    )

    series = [
        ("ret_mkt",     "Market (Mkt-RF)", _ATTR_COLORS["mkt"]),
        ("ret_smb",     "Size (SMB)",      _ATTR_COLORS["smb"]),
        ("ret_hml",     "Value (HML)",     _ATTR_COLORS["hml"]),
        ("ret_alpha",   "Alpha",           _ATTR_COLORS["alpha"]),
        ("idio_return", "Idiosyncratic",   _ATTR_COLORS["idio"]),
    ]

    fig = go.Figure()
    for col, label, color in series:
        fig.add_trace(go.Scatter(
            x=pct.index,
            y=pct[col],
            name=label,
            mode="lines",
            stackgroup="one",
            fillcolor=_hex_rgba(color, 0.75),
            line=dict(width=0.5, color=color),
            hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        font=dict(family="IBM Plex Mono"),
        title=dict(
            text=(
                f"{portfolio_name};  Return Attribution by Factor "
                f"(% of |total|, {smoothing_days}d smoothed, Rolling {window_size}d FF3)"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Share of Total Return", tickformat=".0%", range=[0, 1]),
        legend=dict(borderwidth=1, font=dict(size=11)),
        height=420,
        margin=dict(l=60, r=40, t=60, b=60),
        hovermode="x unified",
    )
    return fig


def build_risk_decomp_chart(
    portfolio_results: pd.DataFrame,
    portfolio_name: str,
    window_size: int,
) -> go.Figure:
    """
    Stacked area chart showing the % share of total variance from each factor
    and idiosyncratic risk (Var = β'Ωβ + σ²_idio decomposition).
    """
    risk_df = portfolio_results[["var_mkt", "var_smb", "var_hml", "idio_var", "total_var"]].copy()
    risk_pct = (
        risk_df[["var_mkt", "var_smb", "var_hml", "idio_var"]]
        .div(risk_df["total_var"], axis=0)
        .fillna(0)
    )

    series = [
        ("var_mkt",  "Market Risk (Mkt-RF)", _ATTR_COLORS["mkt"]),
        ("var_smb",  "Size Risk (SMB)",      _ATTR_COLORS["smb"]),
        ("var_hml",  "Value Risk (HML)",     _ATTR_COLORS["hml"]),
        ("idio_var", "Idiosyncratic Risk",   _ATTR_COLORS["idio"]),
    ]

    fig = go.Figure()
    for col, label, color in series:
        fig.add_trace(go.Scatter(
            x=risk_pct.index,
            y=risk_pct[col],
            name=label,
            mode="lines",
            stackgroup="one",
            fillcolor=_hex_rgba(color, 0.75),
            line=dict(width=0.5, color=color),
            hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        font=dict(family="IBM Plex Mono"),
        title=dict(
            text=(
                f"{portfolio_name}; Risk Decomposition"
                f"(Rolling {window_size}d FF3)"
            ),
            font=dict(size=14),
        ),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Share of Total Variance", tickformat=".0%", range=[0, 1]),
        legend=dict(borderwidth=1, font=dict(size=11)),
        height=420,
        margin=dict(l=60, r=40, t=60, b=60),
        hovermode="x unified",
    )
    return fig


def build_rolling_params_chart(
    portfolio_results: pd.DataFrame,
    portfolio_name: str,
    window_size: int,
) -> go.Figure:
    """
    Four-panel chart with rolling Market beta, SMB beta, HML beta, and Alpha.
    """
    params = [
        ("beta_mkt", "Market Beta (Mkt-RF)", _ATTR_COLORS["mkt"],   1.0),
        ("beta_smb", "Size Beta (SMB)",       _ATTR_COLORS["smb"],   0.0),
        ("beta_hml", "Value Beta (HML)",      _ATTR_COLORS["hml"],   0.0),
        ("alpha",    "Alpha",                 _ATTR_COLORS["alpha"], 0.0),
    ]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[label for _, label, _, _ in params],
    )

    for row_idx, (col, label, color, hline_val) in enumerate(params, start=1):
        fig.add_trace(go.Scatter(
            x=portfolio_results.index,
            y=portfolio_results[col],
            name=label,
            mode="lines",
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.4f}}<extra></extra>",
        ), row=row_idx, col=1)

        fig.add_hline(
            y=hline_val,
            line=dict(color="#666666", dash="dot", width=0.8),
            row=row_idx, col=1,
        )

    # Format alpha axis as percentage
    fig.update_yaxes(tickformat=".2%", row=4, col=1)

    fig.update_layout(
        font=dict(family="IBM Plex Mono"),
        title=dict(
            text=f"{portfolio_name}; Rolling FF3 Parameters ({window_size}d window)",
            font=dict(size=14),
        ),
        height=860,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=60, r=40, t=80, b=60),
    )
    return fig


def render_multifactor_attribution_section(results) -> None:
    """
    Render the full FF3 attribution section: static-param metrics + 4 charts.

    Parameters
    ----------
    results : MultiFactorAttributionResults
    """
    # st.markdown("#### Full-Sample Factor Loadings")
    # p = results.static_params
    # c1, c2, c3, c4 = st.columns(4)
    # with c1:
    #     st.metric("Alpha (daily)", f"{p['const']:.4%}")
    # with c2:
    #     st.metric("Market Beta", f"{p['Mkt-RF']:.3f}")
    # with c3:
    #     st.metric("Size Beta (SMB)", f"{p['SMB']:.3f}")
    # with c4:
    #     st.metric("Value Beta (HML)", f"{p['HML']:.3f}")

    pr = results.portfolio_results
    # st.dataframe(pr)

    st.plotly_chart(
        build_cumulative_decomp_chart(pr, results.portfolio_name, results.window_size),
        use_container_width=True,
    )
    st.plotly_chart(
        build_return_attribution_chart(pr, results.portfolio_name, results.window_size),
        use_container_width=True,
    )
    st.plotly_chart(
        build_risk_decomp_chart(pr, results.portfolio_name, results.window_size),
        use_container_width=True,
    )
    st.plotly_chart(
        build_rolling_params_chart(pr, results.portfolio_name, results.window_size),
        use_container_width=True,
    )
