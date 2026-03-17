"""
optimizations.py
----------------
Constrained mean-variance optimization via numerical methods (SLSQP).
Long-only (weights in [0, 1]), fully-invested (weights sum to 1).
No I/O, no UI dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio_construction.frontier import port_return, port_volatility


# Alias to match the formula convention used internally
def _port_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return port_volatility(w, cov)


# ── Single-portfolio solvers ──────────────────────────────────────────────────

def minimize_vol(target_return: float, er: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Minimum-volatility portfolio subject to:
      - weights in [0, 1]
      - weights sum to 1
      - portfolio return == target_return
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {
            "type": "eq",
            "args": (er,),
            "fun": lambda w, er: target_return - port_return(w=w, r=er),
        },
    )
    result = minimize(
        _port_vol, init_guess,
        args=(cov,),
        method="SLSQP",
        options={"disp": False},
        constraints=constraints,
        bounds=bounds,
    )
    return result.x


def max_sharpe_constrained(er: np.ndarray, cov: np.ndarray, rfr: float) -> np.ndarray:
    """
    Maximum-Sharpe portfolio subject to:
      - weights in [0, 1]
      - weights sum to 1
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    def neg_sharpe(w: np.ndarray, rfr: float, er: np.ndarray, cov: np.ndarray) -> float:
        r = port_return(w, er)
        v = _port_vol(w, cov)
        return -(r - rfr) / v

    result = minimize(
        neg_sharpe, init_guess,
        args=(rfr, er, cov),
        method="SLSQP",
        options={"disp": False},
        constraints=constraints,
        bounds=bounds,
    )
    return result.x


def gmv_constrained(cov: np.ndarray) -> np.ndarray:
    """
    Global Minimum Variance portfolio (constrained).
    Trick: maximise Sharpe with all-ones expected returns and rfr=0,
    which is equivalent to minimising variance.
    """
    n = cov.shape[0]
    return max_sharpe_constrained(er=np.ones(n), cov=cov, rfr=0.0)


# ── Frontier curve ────────────────────────────────────────────────────────────

def optimal_weights(
    er: np.ndarray,
    cov: np.ndarray,
    n_points: int,
    min_ret: float | None = None,
) -> list[np.ndarray]:
    """
    Solve n_points minimum-vol portfolios spanning [min_ret, er.max()].
    Defaults min_ret to er.min(), but pass the constrained GMV return to
    restrict output to the efficient (upper) half of the frontier.
    """
    lo = er.min() if min_ret is None else min_ret
    target_rets = np.linspace(lo, er.max(), num=n_points)
    return [minimize_vol(target_return=r, er=er, cov=cov) for r in target_rets]


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ConstrainedResults:
    # Global Minimum Variance (constrained)
    w_gmv: np.ndarray
    gmv_ret: float
    gmv_std: float

    # Max-Sharpe (constrained)
    w_tan: np.ndarray
    tan_ret: float
    tan_std: float

    # Efficient frontier curve (sorted by vol for clean plotting)
    frontier_vols: np.ndarray
    frontier_rets: np.ndarray


# ── Main computation ──────────────────────────────────────────────────────────

def compute_constrained_frontier(
    miu: pd.Series,
    sigma: pd.DataFrame,
    rf_rate: float,
    n_points: int = 65,
) -> ConstrainedResults:
    """
    Compute the constrained (long-only) efficient frontier and key portfolios.

    Parameters
    ----------
    miu      : Series of mean monthly returns (length n).
    sigma    : Covariance matrix DataFrame (n × n).
    rf_rate  : Monthly risk-free rate.
    n_points : Number of points on the constrained frontier curve.

    Returns
    -------
    ConstrainedResults dataclass.
    """
    er  = miu.values
    cov = sigma.values

    # GMV
    w_gmv   = gmv_constrained(cov)
    gmv_ret = port_return(w_gmv, er)
    gmv_std = _port_vol(w_gmv, cov)

    # Max Sharpe
    w_tan   = max_sharpe_constrained(er, cov, rf_rate)
    tan_ret = port_return(w_tan, er)
    tan_std = _port_vol(w_tan, cov)

    # Frontier curve — span from gmv_ret upward so only the efficient
    # (upper) half of the parabola is traced, avoiding the zigzag pattern
    # that arises when lower-half points sort to the same vol range.
    target_rets   = np.linspace(gmv_ret, er.max(), num=n_points)
    ws            = [minimize_vol(target_return=r, er=er, cov=cov) for r in target_rets]
    raw_rets      = np.array([port_return(w, er) for w in ws])
    raw_vols      = np.array([_port_vol(w, cov)  for w in ws])

    # Sort by vol for a clean left-to-right line
    order          = np.argsort(raw_vols)
    frontier_vols  = raw_vols[order]
    frontier_rets  = raw_rets[order]

    return ConstrainedResults(
        w_gmv=w_gmv, gmv_ret=gmv_ret, gmv_std=gmv_std,
        w_tan=w_tan, tan_ret=tan_ret, tan_std=tan_std,
        frontier_vols=frontier_vols, frontier_rets=frontier_rets,
    )
