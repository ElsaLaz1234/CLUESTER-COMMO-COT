"""
Full analysis pipeline — data loading, clustering, metrics, plots.
"""
import warnings
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR
from data_loader import load_prices, load_cot, compute_log_returns
from clustering import (
    compute_cot_clusters,
    compute_return_clusters,
    build_combined_cot_clusters,
)
from metrics import (
    compute_lead_lag,
    summarise_lead_lag,
    lead_lag_by_subperiod,
    compare_categories,
    compute_cluster_stability,
    ari_at_transition_weeks,
)
import plots

warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(refresh: bool = False):
    """
    Load prices, COT, and derive the canonical weekly date grid.

    Returns
    -------
    (prices, log_ret, cot, dates)
    """
    print("Loading price data...")
    prices  = load_prices(refresh=refresh)
    log_ret = compute_log_returns(prices)

    print("Loading COT data...")
    cot = load_cot(refresh=refresh)
    print(f"  COT rows: {len(cot):,}  |  "
          f"{cot['date'].min().date()} – {cot['date'].max().date()}")

    # Canonical weekly grid: price dates after 26-week warm-up
    dates = log_ret.dropna(how="all").index
    dates = dates[dates >= "2007-04-01"]  # 52-week (12-month) warm-up from 2006-01-01

    return prices, log_ret, cot, dates


def run(refresh: bool = False, make_plots: bool = True) -> None:
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    _, log_ret, cot, dates = load_data(refresh=refresh)

    # --- Return clusters ---
    print("Computing return clusters...")
    ret_clusters = compute_return_clusters(log_ret, dates)
    print(f"  {ret_clusters['date'].nunique()} weeks")

    # --- COT clusters: 4 individual categories + combined ---
    versions = {"M1_MM": "MM", "M2_PM": "PM", "M3_SD": "SD", "M4_OR": "OR"}
    category_lead_lag: dict[str, pd.DataFrame] = {}

    for version, cat in versions.items():
        print(f"Computing COT clusters [{version}]...")
        cot_clust = compute_cot_clusters(cot, dates, category=cat)
        if cot_clust.empty:
            print(f"  WARNING: no clusters for {version}, skipping")
            continue
        category_lead_lag[version] = compute_lead_lag(cot_clust, ret_clusters)

    print("Computing COT clusters [M5_combined]...")
    combined = build_combined_cot_clusters(cot, dates)
    if not combined.empty:
        category_lead_lag["M5_combined"] = compute_lead_lag(combined, ret_clusters)

    if not category_lead_lag:
        raise RuntimeError("No lead-lag results — check COT data.")

    primary = "M1_MM" if "M1_MM" in category_lead_lag else next(iter(category_lead_lag))
    ll = category_lead_lag[primary]

    # --- Summaries ---
    summary       = summarise_lead_lag(ll)
    subperiod_df  = lead_lag_by_subperiod(ll)
    cat_pivot     = compare_categories(category_lead_lag)
    stability     = compute_cluster_stability(ret_clusters)
    transition_ari = ari_at_transition_weeks(ll, stability)

    print(f"\n=== Lead-lag summary ({primary}) ===")
    print(summary.to_string())
    print("\n=== ARI by COT category ===")
    print(cat_pivot.to_string())
    print("\n=== ARI: transition vs stable weeks ===")
    print(transition_ari.to_string())

    # --- Save tables ---
    r = RESULTS_DIR
    summary.to_csv(f"{r}/ari_summary_MM.csv")
    cat_pivot.to_csv(f"{r}/ari_by_category.csv")
    stability.to_csv(f"{r}/cluster_stability.csv", index=False)
    transition_ari.to_csv(f"{r}/ari_transition_vs_stable.csv")
    ll.to_csv(f"{r}/lead_lag_MM_full.csv", index=False)
    ret_clusters.to_csv(f"{r}/return_clusters.csv", index=False)
    if not subperiod_df.empty:
        subperiod_df.to_csv(f"{r}/ari_subperiods.csv", index=False)

    # --- Plots ---
    if make_plots:
        print("\nGenerating plots...")
        plots.plot_ari_by_lag(summary, title=f"Mean ARI by lag ({primary})", fname="ari_by_lag")
        plots.plot_lift_by_lag(summary)
        plots.plot_ari_timeseries(ll, lag=4, fname="ari_timeseries_h1m")
        plots.plot_subperiod_heatmap(subperiod_df)
        plots.plot_category_comparison(cat_pivot)
        plots.plot_stability(stability)
        plots.plot_transition_ari(transition_ari)
        plots.plot_cluster_labels_heatmap(ret_clusters, "Return clusters over time", "return_clusters_heatmap")
        if "M1_MM" in category_lead_lag:
            cot_mm = compute_cot_clusters(cot, dates, category="MM")
            plots.plot_cluster_labels_heatmap(cot_mm, "COT MM clusters over time", "cot_mm_clusters_heatmap")

    print(f"\nDone. Results in ./{RESULTS_DIR}/")
