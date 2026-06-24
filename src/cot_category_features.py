"""
M1–M5 COT category feature matrices.
M5_combined concatenates all 4 categories → (9, 12) feature matrix.
"""
import numpy as np
import pandas as pd

from config import TICKERS_SHORT as TICKERS, COT_CATEGORIES
from src.features import build_cot_feature_matrix
from src.clustering import _cluster_one_week, compute_cot_clusters
from config import COPHENETIC_THRESHOLD


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
