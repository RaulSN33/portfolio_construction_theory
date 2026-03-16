"""
performance_attribution.py
--------------------------
Pure-math layer for single-factor (CAPM) rolling performance attribution.

Computes:
  - Excess returns (vs GS10 risk-free rate from FRED)
  - Static OLS regression (full-sample alpha, beta, R²)
  - Rolling OLS regression (alpha, beta, r_squared, variance decomposition)
  - Return decomposition (actual, factor, idiosyncratic)
  - Cumulative sums for plotting
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from fredapi import Fred
import streamlit as st

MARKET_TICKER = "^GSPC"


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class AttributionResults:
    stock: str
    window_size: int

    # Static OLS (full sample)
    static_alpha: float
    static_beta: float
    static_r2: float

    # Excess returns used for the scatter chart (stock + market columns)
    excess_returns: pd.DataFrame

    # Rolling regression results indexed by date
    # columns: alpha, beta, factor_return, total_var,
    #          systematic_var, idiosyncratic_var, r_squared
    rolling: pd.DataFrame

    # Return decomposition (actual_return, factor_return, idio_return)
    return_decomp: pd.DataFrame

    # Cumulative sums of return_decomp (for the decomposition line chart)
    cumsum_decomp: pd.DataFrame


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_excess_returns(
    stock: str,
    start_date: str,
    end_date: str,
    fred_api_key: str,
) -> pd.DataFrame:
    """
    Download stock + ^GSPC monthly prices and subtract the GS10 risk-free rate.

    Returns a DataFrame with columns [stock, "^GSPC"] containing monthly
    excess returns over the requested date range.
    """
    tickers = [stock, MARKET_TICKER]
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval="1mo",
        progress=False,
        auto_adjust=True,
    )["Close"]

    if isinstance(raw, pd.Series):
        raise ValueError(
            f"Could not download data for both '{stock}' and '{MARKET_TICKER}'. "
            "Check the ticker symbol and date range."
        )

    returns = raw.pct_change().dropna().iloc[:-1]
    returns = returns.resample("ME").last()

    if len(returns) < 2:
        raise ValueError("Not enough monthly observations after download. Widen the date range.")

    # Risk-free rate from FRED (GS10: 10-year treasury annual yield → monthly)
    fred = Fred(api_key=fred_api_key)
    rfr_annual = fred.get_series("GS10") / 100
    rfr_monthly = (
        rfr_annual
        .resample("ME").last()
        .loc[returns.index[0]: returns.index[-1]]
        .apply(lambda v: (1 + v) ** (1 / 12) - 1)
        .ffill()
    )

    excess = returns.sub(rfr_monthly, axis="index").dropna()
    return excess


# ── Rolling OLS ───────────────────────────────────────────────────────────────

def _run_rolling_ols(
    y: pd.Series,
    X: pd.DataFrame,
    window_size: int,
    market_col: str,
) -> pd.DataFrame:
    """
    Fit OLS on each rolling window of length `window_size`.

    For each window ending at date `day`, stores:
      alpha, beta           — regression parameters
      factor_return         — fitted value at the last observation
      total_var             — variance of stock excess returns in window
      systematic_var        — variance explained by the factor
      idiosyncratic_var     — variance of OLS residuals (unexplained)
      r_squared             — R² of the model
    """
    records: dict = {}
    range_total=len(X) - window_size + 1
    print(f'total i {range_total}')
    for i in range(range_total):
        day = X.index[i + window_size - 1]
        window_X = X.iloc[i: i + window_size]
        # print(f'{i}')
        # print(f'type de y {type(y)}')
        window_y = y.iloc[i: i + window_size]
        # print('Aqui no fue 3.1')
        model = sm.OLS(window_y, window_X).fit()
        # print('Aqui no fue 3.2')

        total_var = float(window_y.var())
        idio_var  = float(model.resid.var())
        # print('Aqui no fue 3.3')
        predictions = model.predict()#.iloc[-1]
        # print(f'{predictions}')

        records[day] = {
            "alpha":             float(model.params["const"]),
            "beta":              float(model.params[market_col]),
            "factor_return":     float(model.predict()[-1]), #!!!!!
            # "factor_return":     float(model.predict().iloc[-1]), #!!!!!
            "total_var":         total_var,
            "systematic_var":    total_var - idio_var,
            "idiosyncratic_var": idio_var,
            "r_squared":         float(model.rsquared),
        }
        # print('Aqui no fue 3.5')

    results = pd.DataFrame(records).T
    # print(results)
    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def compute_attribution(
    stock: str,
    start_date: str,
    end_date: str,
    window_size: int,
    fred_api_key: str,
) -> AttributionResults:
    """
    Full attribution pipeline:
      download → excess returns → static OLS → rolling OLS → decomposition.

    Parameters
    ----------
    stock       : Ticker symbol (e.g. "QQQ")
    start_date  : ISO date string "YYYY-MM-DD"
    end_date    : ISO date string "YYYY-MM-DD"
    window_size : Rolling window length in months
    fred_api_key: FRED API key for GS10 series

    Returns
    -------
    AttributionResults dataclass with all results pre-computed.
    """
    excess = fetch_excess_returns(stock, start_date, end_date, fred_api_key)
    print('aqui no 1')
    if len(excess) <= window_size:
        raise ValueError(
            f"Not enough data for a {window_size}-month rolling window. "
            f"Only {len(excess)} months available after applying the risk-free rate. "
            "Widen the date range or reduce the window size."
        )

    y = excess[stock]
    X = sm.add_constant(excess[MARKET_TICKER])

    # Static OLS (full sample)
    static_model = sm.OLS(y, X).fit()
    # print('aqui no 2')
    # Rolling OLS
    rolling = _run_rolling_ols(y, X, window_size, MARKET_TICKER)
    # print('aqui no 3')
    # Return decomposition
    # st.print(y)
    actual = y.iloc[window_size - 1:].rename("actual_return")
    return_decomp = pd.DataFrame({
        "actual_return": actual,
        "factor_return": rolling["factor_return"],
    }).dropna()
    return_decomp["idio_return"] = (
        return_decomp["actual_return"] - return_decomp["factor_return"]
    )

    return AttributionResults(
        stock=stock,
        window_size=window_size,
        static_alpha=float(static_model.params["const"]),
        static_beta=float(static_model.params[MARKET_TICKER]),
        static_r2=float(static_model.rsquared),
        excess_returns=excess,
        rolling=rolling,
        return_decomp=return_decomp,
        cumsum_decomp=return_decomp.cumsum(),
    )
