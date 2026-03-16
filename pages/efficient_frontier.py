"""
pages/efficient_frontier.py
----------------------------
Efficient Frontier page — logic extracted from the original __main__.py.
Page config and CSS are handled once by the multi-page router (__main__.py).
"""
import pandas as pd
import streamlit as st

from src.backend.backend import DownloadError, load_market_data, parse_tickers
from src.dashboard.dashboard import (
    render_analysis_controls,
    render_chart,
    render_header,
    render_metrics,
    render_sidebar,
    render_tables,
)
from src.portfolio_construction.frontier import compute_efficient_frontier


# ── Sidebar inputs ────────────────────────────────────────────────────────────
inputs = render_sidebar()

# ── Static header ─────────────────────────────────────────────────────────────
render_header()


# ── Cached download ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _download(tickers_tuple: tuple, start: str, end: str):
    """Thin cached wrapper around load_market_data."""
    return load_market_data(list(tickers_tuple), start, end)


# ── Trigger a new download only when "Run Analysis" is clicked ───────────────
if inputs["run"]:
    stocks = parse_tickers(inputs["tickers_raw"])
    if len(stocks) < 2:
        st.error("Please enter at least 2 tickers.")
        st.stop()

    with st.spinner(f"Downloading data for: {', '.join(stocks)}…"):
        try:
            prices, returns, valid_stocks = _download(
                tuple(stocks),
                str(inputs["start_date"]),
                str(inputs["end_date"]),
            )
        except DownloadError as e:
            st.error(f"Download failed: {e}")
            st.stop()
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.session_state["prices"]           = prices
    st.session_state["returns"]          = returns
    st.session_state["valid_stocks"]     = valid_stocks
    st.session_state["stocks_requested"] = stocks


# ── Wait for first run ────────────────────────────────────────────────────────
if "prices" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **▶ Run Analysis** to begin.")
    st.stop()

# ── Restore persisted data ────────────────────────────────────────────────────
prices       = st.session_state["prices"]
returns      = st.session_state["returns"]
valid_stocks = st.session_state["valid_stocks"]

removed = set(st.session_state.get("stocks_requested", [])) - set(valid_stocks)
if removed:
    st.warning(f"Removed tickers with no data: {', '.join(sorted(removed))}")

# ── Analysis controls (no download triggered) ─────────────────────────────────
analysis = render_analysis_controls(prices.index)

mask = (
    (returns.index >= pd.Timestamp(analysis["start"]))
    & (returns.index <= pd.Timestamp(analysis["end"]))
)
returns_filtered = returns.loc[mask]

if returns_filtered.shape[0] < 6:
    st.error("Analysis window too narrow — need at least 6 monthly observations.")
    st.stop()

# ── Compute efficient frontier on the filtered window ─────────────────────────
miu     = returns_filtered.mean()
sigma   = returns_filtered.cov()
std_dev = returns_filtered.std()
rf_rate = analysis["rf_rate"]

results = compute_efficient_frontier(miu, sigma, rf_rate=rf_rate)

# ── Render results ────────────────────────────────────────────────────────────
render_metrics(valid_stocks, returns_filtered, results, rf_rate=rf_rate)
render_tables(valid_stocks, results, rf_rate=rf_rate)
render_chart(results, miu, std_dev, valid_stocks, rf_rate=rf_rate)
