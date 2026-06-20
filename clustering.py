"""
Hierarchical clustering (Ward / correlation distance) applied weekly
to both COT positioning features and return features.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform

from config import (
    TICKERS_SHORT as TICKERS, ROLLING_WINDOW, RETURN_LAGS,
    N_CLUSTERS, COPHENETIC_THRESHOLD, COT_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Z-score normalisation (rolling 26 weeks)
# ---------------------------------------------------------------------------

def rolling_zscore(series: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    mu = series.rolling(window, min_periods=window // 2).mean()
    sigma = series.rolling(window, min_periods=window // 2).std()
    return (series - mu) / sigma.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Core clustering routine
# ---------------------------------------------------------------------------

def _cluster_one_week(feature_matrix: np.ndarray, n_clusters: int = N_CLUSTERS):
    """
    feature_matrix : (n_assets, n_features)
    Returns (labels, cophenetic_corr).
    Labels are 0-indexed integers.
    """
    n = feature_matrix.shape[0]
    if n < n_clusters:
        return np.zeros(n, dtype=int), np.nan

    # Correlation matrix → distance matrix
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(feature_matrix)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    dist = 1 - corr
    np.fill_diagonal(dist, 0.0)

    # Ward linkage on condensed distance matrix
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    # Cophenetic correlation
    c, _ = cophenet(Z, condensed)

    labels = fcluster(Z, n_clusters, criterion="maxclust") - 1  # 0-indexed
    return labels, float(c)


# ---------------------------------------------------------------------------
# COT positioning clusters
# ---------------------------------------------------------------------------

def build_cot_feature_matrix(
    cot_df: pd.DataFrame,
    date: pd.Timestamp,
    category: str = "MM",
) -> np.ndarray | None:
    """
    Returns (9, 3) matrix: [z_net, z_delta_net, z_com_mm_ratio] per ticker.
    Uses the ROLLING_WINDOW window ending at `date`.
    """
    sub = cot_df[(cot_df["category"] == category) & (cot_df["date"] <= date)].copy()
    sub = sub.sort_values("date")

    result = []
    for ticker in TICKERS:
        ts = sub[sub["ticker"] == ticker].set_index("date")[["net", "delta_net", "com_mm_ratio"]]
        ts = ts[~ts.index.duplicated()].sort_index()
        window = ts.tail(ROLLING_WINDOW)
        if len(window) < ROLLING_WINDOW // 2:
            return None  # insufficient history

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


def compute_cot_clusters(
    cot_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    category: str = "MM",
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: date, ticker, label, cophenetic, flagged.
    """
    records = []
    for date in dates:
        feat = build_cot_feature_matrix(cot_df, date, category=category)
        if feat is None or np.isnan(feat).all():
            continue
        feat = np.nan_to_num(feat, nan=0.0)
        labels, coph = _cluster_one_week(feat)
        flagged = (not np.isnan(coph)) and (coph < COPHENETIC_THRESHOLD)
        for i, ticker in enumerate(TICKERS):
            records.append({
                "date": date,
                "ticker": ticker,
                "label": int(labels[i]),
                "cophenetic": coph,
                "flagged": flagged,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Return co-movement clusters
# ---------------------------------------------------------------------------

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

    window = log_returns.iloc[idx - n_lags + 1: idx + 1]  # (n_lags, 9)
    if window.shape[0] < n_lags:
        return None

    cols = [c.replace("=F", "") for c in log_returns.columns]
    ordered = [window[c] if c in window.columns else window.iloc[:, 0] * np.nan for c in TICKERS]
    matrix = np.column_stack([s.values for s in ordered]).T  # (9, n_lags)
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix, nan=0.0)
    return matrix


def compute_return_clusters(
    log_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    records = []
    for date in dates:
        feat = build_return_feature_matrix(log_returns, date)
        if feat is None:
            continue
        labels, coph = _cluster_one_week(feat)
        flagged = (not np.isnan(coph)) and (coph < COPHENETIC_THRESHOLD)
        for i, ticker in enumerate(TICKERS):
            records.append({
                "date": date,
                "ticker": ticker,
                "label": int(labels[i]),
                "cophenetic": coph,
                "flagged": flagged,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Combined (M5) — all 4 COT categories concatenated
# ---------------------------------------------------------------------------

def build_combined_cot_clusters(
    cot_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Cluster using all 4 COT category feature matrices concatenated → (9, 12)."""
    cats = list(COT_CATEGORIES.keys())
    records = []
    for date in dates:
        parts = []
        for cat in cats:
            feat = build_cot_feature_matrix(cot_df, date, category=cat)
            if feat is None:
                break
            parts.append(feat)
        if len(parts) < len(cats):
            continue
        combined = np.nan_to_num(np.hstack(parts), nan=0.0)
        labels, coph = _cluster_one_week(combined)
        flagged = (not np.isnan(coph)) and (coph < COPHENETIC_THRESHOLD)
        for i, ticker in enumerate(TICKERS):
            records.append({
                "date": date, "ticker": ticker,
                "label": int(labels[i]), "cophenetic": coph, "flagged": flagged,
            })
    return pd.DataFrame(records)
