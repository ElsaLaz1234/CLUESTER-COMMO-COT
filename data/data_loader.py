"""
Download and cache weekly price data from Yahoo Finance.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import ALL_TICKERS, TICKERS_SHORT, START_DATE, END_DATE, DATA_DIR

Path(DATA_DIR).mkdir(exist_ok=True)


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Return weekly close prices for all tickers (one column per ticker, short name)."""
    cache = Path(DATA_DIR) / "prices_weekly.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = {}
    for ticker in ALL_TICKERS:
        short = ticker.replace("=F", "")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, actions=False)
            if hist.empty:
                warnings.warn(f"No data for {ticker}")
                continue
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            weekly = hist["Close"].resample("W-FRI").last().dropna()
            frames[short] = weekly
        except Exception as e:
            warnings.warn(f"Failed to download {ticker}: {e}")

    if not frames:
        raise RuntimeError("No price data downloaded — check internet connection.")

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices.to_parquet(cache)
    print(f"  Prices: {prices.shape[1]} tickers, {len(prices)} weeks "
          f"({prices.index[0].date()} – {prices.index[-1].date()})")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))
