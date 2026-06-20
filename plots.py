"""
Visualisation utilities.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR, LAGS, SUBPERIODS, LAG_LABELS

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def _save(fig, name: str):
    path = Path(RESULTS_DIR) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------

def plot_ari_by_lag(summary_df: pd.DataFrame, title: str = "ARI by lag", fname: str = "ari_by_lag"):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(summary_df))
    ax.bar(x, summary_df["mean_ari"], color="steelblue", alpha=0.8, label="Mean ARI")
    ax.axhline(summary_df["mean_ari"].iloc[0], color="crimson", ls="--", lw=1.2, label=LAG_LABELS.get(0, "h=0 (sync)"))
    ax.set_xticks(list(x))
    ax.set_xticklabels([LAG_LABELS.get(l, str(l)) for l in summary_df.index])
    ax.set_xlabel("Lag")
    ax.set_ylabel("Mean ARI")
    ax.set_title(title)
    ax.legend()
    _save(fig, fname)


def plot_lift_by_lag(summary_df: pd.DataFrame, fname: str = "lift_by_lag"):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(summary_df))
    colors = ["tomato" if v < 0 else "mediumseagreen" for v in summary_df["lift_ari"]]
    ax.bar(x, summary_df["lift_ari"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LAG_LABELS.get(l, str(l)) for l in summary_df.index])
    ax.set_xlabel("Lag")
    ax.set_ylabel("Lift  (ARI_h − ARI_0)")
    ax.set_title("Predictive lift of COT positioning clusters")
    _save(fig, fname)


def plot_ari_timeseries(lead_lag_df: pd.DataFrame, lag: int = 1, fname: str = "ari_timeseries"):
    sub = lead_lag_df[lead_lag_df["lag"] == lag].set_index("date")["ari"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sub.index, sub.values, lw=0.8, color="steelblue", alpha=0.7)
    ax.axhline(sub.mean(), color="crimson", ls="--", lw=1.2, label=f"Mean = {sub.mean():.3f}")
    ax.set_title(f"ARI(COT_{{t-{LAG_LABELS.get(lag, f'h={lag}')}}}, ret_t) over time")
    ax.set_ylabel("ARI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, fname)


def plot_subperiod_heatmap(subperiod_df: pd.DataFrame, fname: str = "ari_subperiods"):
    if subperiod_df.empty:
        return
    pivot = subperiod_df.pivot(index="subperiod", columns="lag", values="ari")
    pivot.columns = [LAG_LABELS.get(c, str(c)) for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Mean ARI by sub-period and lag")
    ax.set_xlabel("Lag")
    _save(fig, fname)


def plot_category_comparison(cat_pivot: pd.DataFrame, fname: str = "ari_categories"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(cat_pivot))
    for col in cat_pivot.columns:
        ax.plot(list(x), cat_pivot[col].values, marker="o", label=col)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LAG_LABELS.get(l, str(l)) for l in cat_pivot.index])
    ax.set_xlabel("Lag")
    ax.set_ylabel("Mean ARI")
    ax.set_title("ARI by COT category and lag")
    ax.legend()
    _save(fig, fname)


def plot_stability(stability_df: pd.DataFrame, fname: str = "cluster_stability"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(stability_df["date"], stability_df["stability"], lw=0.8, color="steelblue")
    transitions = stability_df[stability_df["transition"]]
    ax.scatter(transitions["date"], transitions["stability"], color="crimson", s=10, zorder=5, label="Transition week")
    ax.set_ylabel("Stability score")
    ax.set_title("Return cluster stability over time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, fname)


def plot_transition_ari(transition_ari: pd.DataFrame, fname: str = "ari_transition_vs_stable"):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    lags = transition_ari.index.tolist()
    x = np.arange(len(lags))
    if "ari_transition" in transition_ari.columns:
        ax.bar(x - width / 2, transition_ari["ari_transition"], width, label="Transition weeks", color="tomato")
    if "ari_stable" in transition_ari.columns:
        ax.bar(x + width / 2, transition_ari["ari_stable"], width, label="Stable weeks", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([LAG_LABELS.get(l, f"h={l}") for l in lags])
    ax.set_ylabel("Mean ARI")
    ax.set_title("ARI: transition periods vs stable periods")
    ax.legend()
    _save(fig, fname)


def plot_cluster_labels_heatmap(clusters_df: pd.DataFrame, title: str, fname: str):
    wide = clusters_df.pivot(index="date", columns="ticker", values="label")
    # Sample at most 200 weeks evenly for readability
    if len(wide) > 200:
        step = len(wide) // 200
        wide = wide.iloc[::step]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        wide.T, cmap="Set1", linewidths=0, ax=ax,
        cbar_kws={"label": "Cluster", "ticks": [0, 1, 2]},
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Ticker")
    date_ticks = np.linspace(0, wide.shape[0] - 1, min(10, wide.shape[0])).astype(int)
    ax.set_xticks(date_ticks)
    ax.set_xticklabels([str(wide.index[i].year) for i in date_ticks], rotation=45)
    _save(fig, fname)
