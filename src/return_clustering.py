"""
Return co-movement cluster stability analysis.
"""
import pandas as pd

from config import LAGS


def compute_cluster_stability(ret_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    s_t = fraction of assets keeping the same label as t-1.
    Flags transition weeks (at least 3 assets change cluster).
    """
    wide = ret_clusters.pivot(index="date", columns="ticker", values="label").sort_index()
    stability = []
    for i in range(1, len(wide)):
        prev = wide.iloc[i - 1].values
        curr = wide.iloc[i].values
        same = (prev == curr).sum()
        total = len(prev)
        n_changed = total - same
        stability.append({
            "date": wide.index[i],
            "stability": same / total,
            "n_changed": n_changed,
            "transition": n_changed >= 3,
        })
    return pd.DataFrame(stability)
