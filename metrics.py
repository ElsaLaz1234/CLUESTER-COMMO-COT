"""
ARI / NMI lead-lag computation, cluster stability, and sub-period breakdowns.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import LAGS, SUBPERIODS, N_CLUSTERS


# ---------------------------------------------------------------------------
# Lead-lag ARI / NMI
# ---------------------------------------------------------------------------

def compute_lead_lag(
    cot_clusters: pd.DataFrame,
    ret_clusters: pd.DataFrame,
    lags: list[int] = LAGS,
) -> pd.DataFrame:
    """
    For each (date, lag h), compute ARI(COT_{t-h}, ret_t).

    Returns DataFrame: date, lag, ari, nmi, n_assets, flagged_cot, flagged_ret
    """
    cot_wide = cot_clusters.pivot(index="date", columns="ticker", values="label")
    ret_wide = ret_clusters.pivot(index="date", columns="ticker", values="label")
    cot_coph = cot_clusters.drop_duplicates("date").set_index("date")["flagged"]
    ret_coph = ret_clusters.drop_duplicates("date").set_index("date")["flagged"]

    dates = ret_wide.index
    records = []

    for date in dates:
        if date not in ret_wide.index:
            continue
        y_ret = ret_wide.loc[date].values
        if np.isnan(y_ret).any():
            continue

        for h in lags:
            # find the COT date h weeks before
            date_h_idx = cot_wide.index.get_indexer([date], method="ffill")[0] - h
            if date_h_idx < 0 or date_h_idx >= len(cot_wide.index):
                continue
            date_h = cot_wide.index[date_h_idx]

            y_cot = cot_wide.loc[date_h].values
            if np.isnan(y_cot).any():
                continue

            ari = adjusted_rand_score(y_ret, y_cot)
            nmi = normalized_mutual_info_score(y_ret, y_cot, average_method="arithmetic")

            records.append({
                "date": date,
                "lag": h,
                "ari": ari,
                "nmi": nmi,
                "n_assets": len(y_ret),
                "flagged_cot": bool(cot_coph.get(date_h, False)),
                "flagged_ret": bool(ret_coph.get(date, False)),
            })

    return pd.DataFrame(records)


def summarise_lead_lag(lead_lag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns mean ARI, NMI, lift vs h=0, per lag.
    """
    agg = (
        lead_lag_df.groupby("lag")[["ari", "nmi"]]
        .mean()
        .rename(columns={"ari": "mean_ari", "nmi": "mean_nmi"})
    )
    agg["lift_ari"] = agg["mean_ari"] - agg.loc[0, "mean_ari"]
    agg["peak"] = agg["mean_ari"] == agg["mean_ari"].max()
    return agg


# ---------------------------------------------------------------------------
# Sub-period analysis
# ---------------------------------------------------------------------------

def lead_lag_by_subperiod(
    lead_lag_df: pd.DataFrame,
    subperiods: dict = SUBPERIODS,
) -> pd.DataFrame:
    records = []
    for name, (start, end) in subperiods.items():
        mask = (lead_lag_df["date"] >= start) & (lead_lag_df["date"] <= end)
        sub = lead_lag_df[mask]
        if sub.empty:
            continue
        agg = sub.groupby("lag")["ari"].mean().reset_index()
        agg["subperiod"] = name
        records.append(agg)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# COT category comparison
# ---------------------------------------------------------------------------

def compare_categories(
    category_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    category_results: {version_name: lead_lag_df}
    Returns pivot: lag x version, values = mean ARI.
    """
    frames = []
    for version, df in category_results.items():
        agg = df.groupby("lag")["ari"].mean().reset_index()
        agg["version"] = version
        frames.append(agg)
    combined = pd.concat(frames, ignore_index=True)
    return combined.pivot(index="lag", columns="version", values="ari")


# ---------------------------------------------------------------------------
# Cluster stability
# ---------------------------------------------------------------------------

def compute_cluster_stability(ret_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    s_t = fraction of assets keeping the same label as t-1.
    Also flags transition weeks (at least 3 assets change cluster).
    """
    wide = ret_clusters.pivot(index="date", columns="ticker", values="label").sort_index()
    stability = []
    for i in range(1, len(wide)):
        prev = wide.iloc[i - 1].values
        curr = wide.iloc[i].values
        same = (prev == curr).sum()
        total = len(prev)
        s = same / total
        n_changed = total - same
        stability.append({
            "date": wide.index[i],
            "stability": s,
            "n_changed": n_changed,
            "transition": n_changed >= 3,
        })
    return pd.DataFrame(stability)


def ari_at_transition_weeks(
    lead_lag_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    lags: list[int] = LAGS,
) -> pd.DataFrame:
    """ARI mean for transition weeks vs stable weeks, per lag."""
    transition_dates = set(stability_df[stability_df["transition"]]["date"])
    lead_lag_df = lead_lag_df.copy()
    lead_lag_df["is_transition"] = lead_lag_df["date"].isin(transition_dates)
    return (
        lead_lag_df.groupby(["lag", "is_transition"])["ari"]
        .mean()
        .unstack("is_transition")
        .rename(columns={True: "ari_transition", False: "ari_stable"})
    )
