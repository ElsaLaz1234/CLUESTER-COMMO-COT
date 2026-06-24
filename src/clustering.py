"""
Hierarchical clustering (Ward, correlation distance) applied weekly
to COT positioning and return feature matrices.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform

from config import TICKERS_SHORT as TICKERS, N_CLUSTERS, COPHENETIC_THRESHOLD
from src.features import build_cot_feature_matrix, build_return_feature_matrix


def _cluster_one_week(feature_matrix: np.ndarray, n_clusters: int = N_CLUSTERS):
    """
    feature_matrix : (n_assets, n_features)
    Returns (labels, cophenetic_corr). Labels are 0-indexed.
    """
    n = feature_matrix.shape[0]
    if n < n_clusters:
        return np.zeros(n, dtype=int), np.nan

    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(feature_matrix)
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)
    dist = 1 - corr
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    c, _ = cophenet(Z, condensed)
    labels = fcluster(Z, n_clusters, criterion="maxclust") - 1
    return labels, float(c)


def compute_cot_clusters(
    cot_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    category: str = "MM",
) -> pd.DataFrame:
    """Returns DataFrame with columns: date, ticker, label, cophenetic, flagged."""
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
                "date": date, "ticker": ticker,
                "label": int(labels[i]), "cophenetic": coph, "flagged": flagged,
            })
    return pd.DataFrame(records)


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
                "date": date, "ticker": ticker,
                "label": int(labels[i]), "cophenetic": coph, "flagged": flagged,
            })
    return pd.DataFrame(records)
