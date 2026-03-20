"""
pages/efficient_frontier.py
----------------------------
Efficient Frontier page — logic extracted from the original __main__.py.
Page config and CSS are handled once by the multi-page router (__main__.py).
"""
from datetime import date

import pandas as pd
import streamlit as st

from PortfolioBacktester.entinties import NaiveBacktest
from PortfolioBacktester.modules.performance_functions import summary_stats

from src.backend.backend import DownloadError, load_market_data, parse_tickers
from src.dashboard.frontier_dashboard import (
    render_analysis_controls,
    render_attribution_controls,
    render_backtest_section,
    render_chart,
    render_header,
    render_metrics,
    render_multifactor_attribution_section,
    render_sidebar,
    render_tables,
)
from src.portfolio_construction.frontier import compute_efficient_frontier
from src.portfolio_construction.multifactor_performance_attribution import (
    compute_multifactor_attribution,
)
from src.portfolio_construction.optimizations import compute_constrained_frontier


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

    today_str = str(date.today())
    with st.spinner(f"Downloading data for: {', '.join(stocks)}…"):
        try:
            prices, returns, valid_stocks = _download(
                tuple(stocks),
                str(inputs["start_date"]),
                today_str,
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

# Use all available history up to the backtesting start date
mask = returns.index <= pd.Timestamp(analysis["end"])
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

with st.spinner("Computing constrained frontier…"):
    constrained = compute_constrained_frontier(miu, sigma, rf_rate=rf_rate)

# ── Render frontier results ───────────────────────────────────────────────────
render_metrics(valid_stocks, returns_filtered, results, rf_rate=rf_rate)
render_tables(valid_stocks, results, rf_rate=rf_rate, constrained=constrained)
render_chart(results, miu, std_dev, valid_stocks, rf_rate=rf_rate, constrained=constrained)

# ── Buy-and-hold backtest for each portfolio ──────────────────────────────────
backtest_start = str(analysis["end"])
backtest_end   = str(date.today())

if backtest_start >= backtest_end:
    st.warning("Backtesting Start must be before today to run the backtest.")
    st.stop()

portfolios_to_backtest = {
    "GMV (Unconstrained)":        results.w_gmv,
    "Max Sharpe (Unconstrained)": results.w_tan,
    "Equal Weight":               results.w_ew,
    "GMV (Constrained)":          constrained.w_gmv,
    "Max Sharpe (Constrained)":   constrained.w_tan,
}

with st.spinner("Running buy-and-hold backtests…"):
    cumulative  = {}
    all_metrics = []
    backtested_daily_weights = {}

    asset_prices_bt = prices[valid_stocks]

    for name, weights in portfolios_to_backtest.items():
        signals_df = pd.DataFrame(
            {pd.Timestamp(backtest_start): weights},
            index=valid_stocks,
        )
        bt = NaiveBacktest(
            start_date=backtest_start,
            end_date=backtest_end,
            signals_df=signals_df,
            asset_prices=asset_prices_bt,
            initial_capital=1,
        )
        bt._run_backtest()
        # bt._asset_returns()
        # st.dataframe(bt.asset_prices)
        # st.dataframe(bt.portfolio_returns)
        cumulative[name] = bt.price_simulation
        cumulative[name].iloc[0] = 1

        metrics = summary_stats(bt.portfolio_returns, periods_per_year=252)
        metrics.index = [name]
        all_metrics.append(metrics)

        # st.markdown(name)
        backtested_daily_weights[name] = bt._reweight_daily_weights(bt.backtested_daily_weights)
        # csv.to_csv(f'{name}.csv')

    metrics_df = pd.concat(all_metrics)
render_backtest_section(cumulative, metrics_df, backtest_start)

# ── Fama-French 3-Factor Attribution ─────────────────────────────────────────
attribution_inputs = render_attribution_controls(list(backtested_daily_weights.keys()))

if attribution_inputs["run"]:
    selected_name    = attribution_inputs["portfolio"]
    window_size      = attribution_inputs["window_size"]
    # weights_array    = backtested_daily_weights[selected_name]
    # selected_weights = pd.Series(weights_array, index=valid_stocks)
    selected_weights = backtested_daily_weights[selected_name]
    # st.dataframe(selected_weights)

    fred_api_key = st.secrets.get("FRED_API_KEY", "")
    if not fred_api_key:
        st.error("FRED_API_KEY not found in .streamlit/secrets.toml")
        st.stop()

    with st.spinner(f"Running FF3 attribution for {selected_name}…"):
        try:
            attr_results = compute_multifactor_attribution(
                portfolio_name=selected_name,
                weights=selected_weights,
                prices=prices,
                window_size=window_size,
                fred_api_key=fred_api_key,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    st.session_state["attr_results"] = attr_results

if "attr_results" in st.session_state:
    render_multifactor_attribution_section(st.session_state["attr_results"])
