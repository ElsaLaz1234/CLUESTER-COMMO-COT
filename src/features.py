"""
COT feature construction and z-score normalization.
"""
import numpy as np
import pandas as pd

from config import TICKERS_SHORT as TICKERS, ROLLING_WINDOW, RETURN_LAGS


def rolling_zscore(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    mu = series.rolling(window, min_periods=window // 2).mean()
    sigma = series.rolling(window, min_periods=window // 2).std()
    return (series - mu) / sigma.replace(0, np.nan)


def build_cot_feature_matrix(
    cot_df: pd.DataFrame,
    date: pd.Timestamp,
    category: str = "MM",
) -> np.ndarray | None:
    """
    Returns (9, 3) matrix: [z_net, z_delta_net, z_com_mm_ratio] per ticker.
    Uses ROLLING_WINDOW weeks ending at `date`.
    """
    sub = cot_df[(cot_df["category"] == category) & (cot_df["date"] <= date)].copy()
    sub = sub.sort_values("date")

    result = []
    for ticker in TICKERS:
        ts = sub[sub["ticker"] == ticker].set_index("date")[["net", "delta_net", "com_mm_ratio"]]
        ts = ts[~ts.index.duplicated()].sort_index()
        window = ts.tail(ROLLING_WINDOW)
        if len(window) < ROLLING_WINDOW // 2:
            return None

        last = window.iloc[-1]
        mu = window.mean()
        sigma = window.std()

        def zscore(val, m, s):
            if s == 0 or np.isnan(s):
                return 0.0
            return float((val - m) / s) if pd.notna(val) else 0.0

        result.append([
            zscore(last["net"], mu["net"], sigma["net"]),
            zscore(last["delta_net"], mu["delta_net"], sigma["delta_net"]),
            zscore(last["com_mm_ratio"], mu["com_mm_ratio"], sigma["com_mm_ratio"]),
        ])

    return np.array(result)  # (9, 3)


def build_return_feature_matrix(
    log_returns: pd.DataFrame,
    date: pd.Timestamp,
    n_lags: int = RETURN_LAGS,
) -> np.ndarray | None:
    """
    Returns (9, n_lags) matrix of raw log returns [r_t, r_{t-1}, ..., r_{t-n+1}].
    """
    idx = log_returns.index.get_indexer([date], method="ffill")[0]
    if idx < n_lags:
        return None

    window = log_returns.iloc[idx - n_lags + 1: idx + 1]
    if window.shape[0] < n_lags:
        return None

    ordered = [window[c] if c in window.columns else window.iloc[:, 0] * np.nan for c in TICKERS]
    matrix = np.column_stack([s.values for s in ordered]).T  # (9, n_lags)
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix, nan=0.0)
    return matrix
