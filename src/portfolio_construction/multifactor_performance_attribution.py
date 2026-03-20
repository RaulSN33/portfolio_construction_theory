"""
multifactor_performance_attribution.py
---------------------------------------
Fama-French 3-factor rolling attribution for a portfolio with time-varying weights.

Steps:
1. Download factor ETF prices (Vanguard style + SPY) to build SMB, HML, Mkt-RF
2. Fetch risk-free rate from FRED (DGS10 → daily)
3. Compute per-stock excess returns from the prices already in session state
4. Run rolling FF3 OLS regression over each stock, storing results per stock per day
5. Aggregate each day's per-stock results using the actual backtested weights for that day
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from fredapi import Fred

# ── Factor ETF universe (Vanguard 3×3 style + SPY as market proxy) ────────────
_FACTOR_TICKERS = ["VTV", "VOE", "VBR", "VV", "VO", "VB", "VUG", "VOT", "VBK", "SPY"]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MultiFactorAttributionResults:
    portfolio_name:    str
    window_size:       int            # rolling window in trading days
    weights:           pd.DataFrame  # time-varying backtested weights (date × ticker)
    portfolio_results: pd.DataFrame  # portfolio-level rolling time series
    excess_returns:    pd.DataFrame  # per-stock excess return history


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_factors(
    portfolio_tickers: list[str],
    prices: pd.DataFrame,
    fred_api_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the FF3 regressor matrix X3 and per-stock excess return DataFrame xret,
    both aligned on a common date index (inner join + dropna).

    Returns
    -------
    X3   : DataFrame [const, Mkt-RF, SMB, HML]
    xret : DataFrame of excess returns, columns = portfolio_tickers
    """
    # ── Factor ETF prices ─────────────────────────────────────────────────────
    factor_prices = yf.download(
        _FACTOR_TICKERS,
        period="10y",
        interval="1d",
        group_by="column",
        progress=False,
        auto_adjust=True,
    )["Close"]
    factor_rets = factor_prices.pct_change().iloc[1:]

    # ── SMB (Small-Minus-Big) ─────────────────────────────────────────────────
    smb = (
        (factor_rets["VBR"] + factor_rets["VB"] + factor_rets["VBK"]) / 3
        - (factor_rets["VTV"] + factor_rets["VV"] + factor_rets["VUG"]) / 3
    )
    smb.name = "SMB"

    # ── HML (High-Minus-Low) ──────────────────────────────────────────────────
    hml = (
        (factor_rets["VTV"] + factor_rets["VBR"]) / 2
        - (factor_rets["VUG"] + factor_rets["VBK"]) / 2
    )
    hml.name = "HML"

    # ── Risk-free rate (FRED DGS10, annual → daily) ───────────────────────────
    fred = Fred(api_key=fred_api_key)
    rfr_annual = fred.get_series("DGS10") / 100
    rfr_annual.index = pd.to_datetime(rfr_annual.index)
    rfr_daily = (
        rfr_annual
        .reindex(smb.index)
        .ffill()
        .apply(lambda r: (1 + r) ** (1 / 262) - 1)
    )

    # ── Market excess return ──────────────────────────────────────────────────
    mkt = factor_rets["SPY"].sub(rfr_daily).rename("Mkt-RF")

    # ── Portfolio stock excess returns ────────────────────────────────────────
    port_rets = prices[portfolio_tickers].pct_change().iloc[1:]
    xret = port_rets.sub(rfr_daily, axis="index")

    # ── Align everything on a common, clean index ─────────────────────────────
    combined = pd.concat([xret, mkt, smb, hml], axis=1).dropna()
    xret = combined[portfolio_tickers]
    mkt  = combined["Mkt-RF"]
    smb  = combined["SMB"]
    hml  = combined["HML"]

    X3 = sm.add_constant(pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": hml}))
    return X3, xret


def _run_rolling_ff3(
    xret: pd.DataFrame,
    X3: pd.DataFrame,
    window_size: int,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Rolling FF3 OLS over all stocks simultaneously. For every window ending at
    date t, store a per-stock DataFrame of attribution & risk components.

    Returns
    -------
    pd.DataFrame with MultiIndex (date, ticker) and columns:
        alpha, beta_mkt, beta_smb, beta_hml,
        ret_alpha, ret_mkt, ret_smb, ret_hml, factor_return, idio_return, actual_return,
        sys_var, idio_var, total_var, var_mkt, var_smb, var_hml
    """
    mkt_col = X3["Mkt-RF"]
    smb_col = X3["SMB"]
    hml_col = X3["HML"]

    daily_stock_results: dict[pd.Timestamp, pd.DataFrame] = {}

    for i in range(len(xret) - window_size + 1):
        day = xret.index[i + window_size - 1]
        w_y = xret.iloc[i : i + window_size]
        w_X = X3.iloc[i : i + window_size]

        model  = sm.OLS(w_y, w_X).fit()
        params = model.params  # shape: (n_factors, n_stocks) when endog is a DataFrame

        # Ensure params is a DataFrame indexed by factor name with stocks as columns
        if not isinstance(params, pd.DataFrame):
            params = pd.DataFrame(params, index=w_X.columns, columns=tickers)
        else:
            params.columns = tickers

        b_mkt = params.loc["Mkt-RF"]
        b_smb = params.loc["SMB"]
        b_hml = params.loc["HML"]
        alpha = params.loc["const"]

        # ── Return attribution at the last observation of the window ──────────
        last_mkt = mkt_col.iloc[i + window_size - 1]
        last_smb = smb_col.iloc[i + window_size - 1]
        last_hml = hml_col.iloc[i + window_size - 1]

        ret_mkt    = b_mkt * last_mkt
        ret_smb    = b_smb * last_smb
        ret_hml    = b_hml * last_hml
        ret_alpha  = alpha
        factor_ret = ret_alpha + ret_mkt + ret_smb + ret_hml
        actual_ret = w_y.iloc[-1]
        idio_ret   = actual_ret - factor_ret

        # ── Risk decomposition: Var(r) = β'Ωβ + σ²_idio ──────────────────────
        w_factors  = w_X[["Mkt-RF", "SMB", "HML"]]
        Omega      = w_factors.cov()                                # (3, 3)
        betas      = np.array([b_mkt.values, b_smb.values, b_hml.values])  # (3, n)

        beta_sigma = betas.T @ Omega.values    # (n, 3)
        contribs   = beta_sigma * betas.T      # (n, 3) per-factor variance contributions

        contribs_df = pd.DataFrame(
            contribs, index=tickers, columns=["var_mkt", "var_smb", "var_hml"]
        )

        resid = model.resid
        if isinstance(resid, pd.DataFrame):
            idio_var = resid.var()
        else:
            idio_var = pd.Series(resid.var(), index=tickers)

        sys_var   = contribs_df.sum(axis=1)
        total_var = sys_var + idio_var

        # ── Store per-stock results for this day ──────────────────────────────
        daily_stock_results[day] = pd.DataFrame({
            "alpha":         alpha,
            "beta_mkt":      b_mkt,
            "beta_smb":      b_smb,
            "beta_hml":      b_hml,
            "ret_alpha":     ret_alpha,
            "ret_mkt":       ret_mkt,
            "ret_smb":       ret_smb,
            "ret_hml":       ret_hml,
            "factor_return": factor_ret,
            "idio_return":   idio_ret,
            "actual_return": actual_ret,
            "sys_var":       sys_var,
            "idio_var":      idio_var,
            "total_var":     total_var,
            "var_mkt":       contribs_df["var_mkt"],
            "var_smb":       contribs_df["var_smb"],
            "var_hml":       contribs_df["var_hml"],
        })  # shape: (n_stocks, n_cols)

    return pd.concat(daily_stock_results)  # MultiIndex: (date, ticker)


def _portfolio_attrib(
    by_stock_attribution: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-stock daily attribution results using the actual backtested
    weights for each day.

    Parameters
    ----------
    by_stock_attribution : MultiIndex DataFrame (date, ticker) × attribution columns
                           — output of _run_rolling_ff3
    weights              : DataFrame (date × ticker) of time-varying portfolio weights
                           — from bt._reweight_daily_weights(bt.backtested_daily_weights)

    Returns
    -------
    DataFrame indexed by date, one portfolio-level row per day
    """
    attribution_dates = by_stock_attribution.index.get_level_values(0).unique()

    rolling_portfolio_results = {}
    for day, w_i in weights.iterrows():
        if day not in attribution_dates:
            continue
        results_i = by_stock_attribution.loc[day]   # (n_tickers, n_cols)
        rolling_portfolio_results[day] = w_i @ results_i   # (n_cols,) weighted sum

    return pd.DataFrame(rolling_portfolio_results).T


# ── Public API ────────────────────────────────────────────────────────────────

def compute_multifactor_attribution(
    portfolio_name: str,
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    window_size: int,
    fred_api_key: str,
) -> MultiFactorAttributionResults:
    """
    Compute Fama-French 3-factor rolling attribution for a portfolio.

    Parameters
    ----------
    portfolio_name : display name used in chart titles
    weights        : time-varying backtested weights, DataFrame (date × ticker)
                     — from bt._reweight_daily_weights(bt.backtested_daily_weights)
    prices         : daily price DataFrame with all portfolio tickers as columns
    window_size    : rolling OLS window in trading days (e.g. 504 = 2 years)
    fred_api_key   : FRED API key for the DGS10 risk-free rate series

    Returns
    -------
    MultiFactorAttributionResults
    """
    tickers = list(weights.columns)

    X3, xret = _build_factors(tickers, prices, fred_api_key)

    if len(xret) < window_size + 1:
        raise ValueError(
            f"Insufficient data: need at least {window_size + 1} trading days "
            f"but only {len(xret)} are available after index alignment."
        )

    # ── Per-stock rolling FF3 ─────────────────────────────────────────────────
    by_stock_attribution = _run_rolling_ff3(xret, X3, window_size, tickers)

    # ── Aggregate with actual backtested weights per day ──────────────────────
    rolling_portfolio_results = _portfolio_attrib(by_stock_attribution, weights)

    return MultiFactorAttributionResults(
        portfolio_name=portfolio_name,
        window_size=window_size,
        weights=weights,
        portfolio_results=rolling_portfolio_results,
        excess_returns=xret,
    )
