"""
Subperiod and regime decomposition of the COT lead-lag signal.
"""
import pandas as pd

from config import SUBPERIODS


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
