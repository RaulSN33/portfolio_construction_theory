"""
frontier.py
-----------
Pure financial-math layer: portfolio statistics and efficient frontier computation.
No I/O, no UI dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Portfolio stat helpers ───────────────────────────────────────────────────

def port_return(w: np.ndarray, r: np.ndarray) -> float:
    return float(w @ r)


def port_volatility(w: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sqrt(w @ sigma @ w))


def sharpe(ret: float, vol: float, rf: float) -> float:
    return (ret - rf) / vol if vol > 0 else float("nan")


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class FrontierResults:
    # Global Minimum Variance
    w_gmv: np.ndarray
    gmv_ret: float
    gmv_std: float

    # Max-Sharpe / Tangency portfolio
    w_tan: np.ndarray
    tan_ret: float
    tan_std: float

    # Equal-weight portfolio
    w_ew: np.ndarray
    ew_ret: float
    ew_std: float

    # Efficient frontier curve (volatility, return pairs)
    frontier_vols: np.ndarray
    frontier_rets: np.ndarray

    # Capital Market Line
    cml_vols: np.ndarray
    cml_rets: np.ndarray
    cml_slope: float


# ── Main computation ─────────────────────────────────────────────────────────

def compute_efficient_frontier(
    miu: pd.Series,
    sigma: pd.DataFrame,
    rf_rate: float,
    n_frontier_pts: int = 65,
    frontier_step: float = 0.001,
) -> FrontierResults:
    """
    Compute the mean-variance efficient frontier and key portfolios.

    Parameters
    ----------
    miu        : Series of mean monthly returns (length n).
    sigma      : Covariance matrix (n × n).
    rf_rate    : Monthly risk-free rate.
    n_frontier_pts : Number of points on the frontier curve.
    frontier_step  : Volatility increment per point along the curve.

    Returns
    -------
    FrontierResults dataclass.
    """
    n = len(miu)
    ones = np.ones(n)
    sigma_arr = sigma.values
    miu_arr = miu.values

    sigma_inv = np.linalg.pinv(sigma_arr)

    # Scalar helpers for the parabola
    a = float(ones @ sigma_inv @ ones)
    b = float(ones @ sigma_inv @ miu_arr)
    c = float(miu_arr @ sigma_inv @ miu_arr)

    # Global Minimum Variance portfolio
    w_gmv = (sigma_inv @ ones) / a
    gmv_ret = port_return(w_gmv, miu_arr)
    gmv_std = port_volatility(w_gmv, sigma_arr)

    # Tangency (Max-Sharpe) portfolio
    excess = miu_arr - rf_rate
    u = np.linalg.solve(sigma_arr, excess)
    w_tan = u / u.sum()
    tan_ret = port_return(w_tan, miu_arr)
    tan_std = port_volatility(w_tan, sigma_arr)

    # Equal-weight portfolio
    w_ew = np.ones(n) / n
    ew_ret = port_return(w_ew, miu_arr)
    ew_std = port_volatility(w_ew, sigma_arr)

    # Efficient frontier curve
    min_vol = np.sqrt(1 / a)
    frontier_vols = np.linspace(min_vol, min_vol + frontier_step * n_frontier_pts, num=n_frontier_pts)
    frontier_rets = b / a + np.sqrt(c - b**2 / a) * np.sqrt(frontier_vols**2 - 1 / a)

    # Capital Market Line
    cml_slope = (tan_ret - rf_rate) / tan_std
    cml_vols = np.linspace(0, frontier_vols.max() * 1.15, 200)
    cml_rets = rf_rate + cml_slope * cml_vols

    return FrontierResults(
        w_gmv=w_gmv, gmv_ret=gmv_ret, gmv_std=gmv_std,
        w_tan=w_tan, tan_ret=tan_ret, tan_std=tan_std,
        w_ew=w_ew, ew_ret=ew_ret, ew_std=ew_std,
        frontier_vols=frontier_vols, frontier_rets=frontier_rets,
        cml_vols=cml_vols, cml_rets=cml_rets, cml_slope=cml_slope,
    )
