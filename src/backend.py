"""
backend.py
----------
Data-fetching and preprocessing layer.
Handles ticker parsing, price download, and return computation.
No UI dependencies.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


# ── Ticker parsing ───────────────────────────────────────────────────────────

def parse_tickers(raw: str) -> list[str]:
    """
    Accept a string of tickers separated by newlines, commas, or spaces.
    Returns a de-duplicated, upper-cased list.
    """
    normalised = raw.replace(",", "\n").replace(" ", "\n")
    seen: set[str] = set()
    result: list[str] = []
    for tok in normalised.splitlines():
        t = tok.strip().upper()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ── Price download ───────────────────────────────────────────────────────────

class DownloadError(Exception):
    """Raised when yfinance fails to return usable price data."""


def fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1mo",
) -> pd.DataFrame:
    """
    Download monthly adjusted close prices from Yahoo Finance.

    Returns
    -------
    DataFrame with one column per valid ticker, indexed by date.

    Raises
    ------
    DownloadError if the download fails or returns no usable data.
    """
    try:
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise DownloadError(str(exc)) from exc

    # Normalise column structure (single vs multi-ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop columns that are entirely NaN
    prices = prices.dropna(axis=1, how="all")

    if prices.empty or prices.shape[1] == 0:
        raise DownloadError("No price data was returned. Check the ticker symbols.")

    return prices


# ── Returns computation ──────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns, dropping the first NaN row and
    any remaining rows with missing values.
    """
    return prices.pct_change().dropna()


# ── Convenience wrapper ──────────────────────────────────────────────────────

def load_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Full pipeline: download → clean → returns.

    Returns
    -------
    prices      : Cleaned price DataFrame.
    returns     : Monthly return DataFrame.
    valid_tickers : Tickers that survived the cleaning step.

    Raises
    ------
    DownloadError if the download fails.
    ValueError   if fewer than 2 valid tickers remain or not enough history.
    """
    prices = fetch_prices(tickers, start_date, end_date)
    prices = prices.dropna()

    valid_tickers = prices.columns.tolist()
    if len(valid_tickers) < 2:
        raise ValueError(
            "Not enough valid tickers after download. Check ticker symbols."
        )

    returns = compute_returns(prices)
    if returns.shape[0] < 12:
        raise ValueError(
            "Not enough monthly data points (need at least 12). Try a wider date range."
        )

    return prices, returns, valid_tickers
