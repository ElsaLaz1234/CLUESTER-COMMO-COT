"""
Dendrogram snapshot analysis — peak/trough signal weeks per sub-period.

Public API
----------
identify_snapshot_weeks(ari_df)  →  snapshot dict
build_feature_matrices(...)      →  (cot_mats, return_mats) dicts keyed by date str
plot_snapshot_dendrograms(...)   →  list of saved paths
plot_crisis_calm_contrast(...)   →  saved path or None
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from config import (
    TICKERS_SHORT as TICKERS,
    SECTOR_COLORS,
    SUBPERIODS,
    CRISIS_PERIODS,
    CALM_PERIODS,
    PERIOD_SLUG,
    RESULTS_DIR,
    SNAPSHOT_LAG,
    LAG_LABELS,
)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

_DENDROGRAMS_DIR = Path(RESULTS_DIR) / "dendrograms"


# ---------------------------------------------------------------------------
# Task 1 — identify snapshot weeks
# ---------------------------------------------------------------------------

def identify_snapshot_weeks(
    ari_df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    For each period in SUBPERIODS find the peak and trough week by ARI at
    lag h=8 (2 months).

    Parameters
    ----------
    ari_df : DataFrame with columns [date, lag, ari].

    Returns
    -------
    {
        "GFC_2008_2009": {
            "peak":   {"date": "2009-11-20", "ARI_h2": 0.68},
            "trough": {"date": "2008-03-14", "ARI_h2": -0.25},
        },
        ...
    }
    """
    df = ari_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    h2 = (
        df[df["lag"] == SNAPSHOT_LAG][["date", "ari"]]
        .drop_duplicates("date")
        .sort_values("date")
    )

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for period, (start, end) in SUBPERIODS.items():
        sub = h2[(h2["date"] >= start) & (h2["date"] <= end)]
        if sub.empty:
            warnings.warn(f"No ARI h={SNAPSHOT_LAG} data for '{period}' — skipping.")
            continue
        result[period] = {
            "peak":   {"date": str(sub.loc[sub["ari"].idxmax(), "date"].date()),
                       "ARI_h2": float(sub["ari"].max())},
            "trough": {"date": str(sub.loc[sub["ari"].idxmin(), "date"].date()),
                       "ARI_h2": float(sub["ari"].min())},
        }

    _print_snapshot_table(result)
    return result


def _print_snapshot_table(snapshots: dict) -> None:
    print("\n" + "=" * 66)
    print(f"SNAPSHOT WEEKS  (ARI at lag h={SNAPSHOT_LAG})")
    print("=" * 66)
    print(f"{'Period':<26}  {'Type':<6}  {'Date':<12}  {'ARI':>8}")
    print("-" * 66)
    for period in list(CRISIS_PERIODS) + list(CALM_PERIODS):
        if period not in snapshots:
            continue
        for kind in ("peak", "trough"):
            info = snapshots[period][kind]
            print(f"{period:<26}  {kind:<6}  {info['date']:<12}  {info['ARI_h2']:>+8.4f}")
    print("=" * 66 + "\n")


# ---------------------------------------------------------------------------
# Task 4 — feature matrix cache
# ---------------------------------------------------------------------------

def build_feature_matrices(
    cot_df: pd.DataFrame,
    log_ret: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Return (cot_mats, return_mats) dicts keyed by "YYYY-MM-DD" date strings.
    Loads from cache if available; otherwise recomputes and saves.
    """
    cot_pkl = Path(RESULTS_DIR) / "cot_mats_w26.pkl"
    ret_pkl = Path(RESULTS_DIR) / "return_mats_w26.pkl"

    if cot_pkl.exists() and ret_pkl.exists():
        print("Loading cached feature matrices...")
        with open(cot_pkl, "rb") as f:
            cot_mats: dict[str, np.ndarray] = pickle.load(f)
        with open(ret_pkl, "rb") as f:
            return_mats: dict[str, np.ndarray] = pickle.load(f)
        print(f"  COT: {len(cot_mats)} obs  |  Return: {len(return_mats)} obs")
        return cot_mats, return_mats

    print("Feature matrices not found — recomputing from raw data. "
          "This may take a few minutes.")
    from clustering import build_cot_feature_matrix, build_return_feature_matrix

    cot_mats, return_mats = {}, {}
    total = len(dates)
    for i, date in enumerate(dates):
        if i % 100 == 0:
            print(f"  {i}/{total}...", end="\r", flush=True)
        key = str(date.date())
        feat = build_cot_feature_matrix(cot_df, date, category="MM")
        if feat is not None:
            cot_mats[key] = np.nan_to_num(feat, nan=0.0)
        feat = build_return_feature_matrix(log_ret, date)
        if feat is not None:
            return_mats[key] = np.nan_to_num(feat, nan=0.0)

    print(f"  Recomputed {len(cot_mats)} COT + {len(return_mats)} return matrices.")
    with open(cot_pkl, "wb") as f:
        pickle.dump(cot_mats, f)
    with open(ret_pkl, "wb") as f:
        pickle.dump(return_mats, f)
    print(f"  Saved → {cot_pkl}\n  Saved → {ret_pkl}")
    return cot_mats, return_mats


# ---------------------------------------------------------------------------
# Dendrogram drawing helpers
# ---------------------------------------------------------------------------

def _ward_linkage(feature_matrix: np.ndarray) -> np.ndarray:
    """Ward linkage from a (9, F) feature matrix using correlation distance."""
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(feature_matrix)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method="ward")


def _draw_panel(ax: plt.Axes, feature_matrix: np.ndarray, title: str) -> None:
    """Draw one dendrogram panel with sector-coloured leaf labels."""
    Z = _ward_linkage(feature_matrix)
    dendrogram(
        Z, labels=TICKERS, ax=ax,
        color_threshold=0,
        above_threshold_color="#666666",
        leaf_rotation=0,
        leaf_font_size=9,
    )
    ax.set_ylabel(r"Distance  $D = 1 - \rho$", fontsize=9)
    ax.set_title(title, fontsize=9, pad=4)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_color(SECTOR_COLORS.get(tick.get_text(), "black"))
        tick.set_fontweight("bold")


def _sector_legend() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color="red",        label="Energy  (CL, NG, RB)"),
        mpatches.Patch(color="darkorange", label="Metals  (GC, SI, HG)"),
        mpatches.Patch(color="green",      label="Agri    (ZC, ZW, ZS)"),
    ]


# ---------------------------------------------------------------------------
# Task 2 — per-period snapshot figures
# ---------------------------------------------------------------------------

def plot_snapshot_dendrograms(
    snapshots: dict[str, dict[str, dict[str, Any]]],
    cot_mats: dict[str, np.ndarray],
    return_mats: dict[str, np.ndarray],
    out_dir: Path | None = None,
) -> list[Path]:
    """
    Save one side-by-side figure per (period, kind) combination.

    Returns list of saved file paths.
    """
    out_dir = out_dir or _DENDROGRAMS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for period, info in snapshots.items():
        slug = PERIOD_SLUG.get(period, period.lower())
        for kind in ("peak", "trough"):
            try:
                path = _save_snapshot_figure(
                    period, slug, kind,
                    info[kind]["date"], info[kind]["ARI_h2"],
                    cot_mats, return_mats, out_dir,
                )
                if path:
                    saved.append(path)
            except Exception as exc:
                warnings.warn(f"{period} {kind}: {exc}")

    return saved


def _save_snapshot_figure(
    period: str, slug: str, kind: str,
    date_str: str, ari_val: float,
    cot_mats: dict[str, np.ndarray],
    return_mats: dict[str, np.ndarray],
    out_dir: Path,
) -> Path | None:
    for label, mats in [("COT", cot_mats), ("return", return_mats)]:
        if date_str not in mats:
            warnings.warn(f"Snapshot {date_str} not found in {label} matrices — skipping.")
            return None

    is_peak  = kind == "peak"
    period_d = period.replace("_", " ")
    kind_label = "peak signal — 12m window ending" if is_peak else "trough signal — 12m window ending"
    lag_display = LAG_LABELS.get(SNAPSHOT_LAG, f"h={SNAPSHOT_LAG}")
    suptitle = (
        f"Dendrograms — {period_d} {kind_label} {date_str}"
        f"  |  ARI {lag_display} = {ari_val:.3f}"
    )
    caption = (
        "Left: COT positioning structure.  Right: Return co-movement 2 months later."
        "  High ARI = similar groupings."
        if is_peak else
        "Low ARI = divergent groupings — COT did not anticipate return co-movement."
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(suptitle, fontsize=10, y=1.01, fontweight="bold")
    _draw_panel(axes[0], cot_mats[date_str],    f"COT features (Ward)\n{date_str}")
    _draw_panel(axes[1], return_mats[date_str], f"Return co-movement (Ward)\n{date_str}")
    fig.legend(handles=_sector_legend(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), fontsize=8, frameon=False)
    plt.tight_layout()

    path = out_dir / f"dendrogram_{slug}_{kind}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Task 3 — crisis vs calm contrast (2×2)
# ---------------------------------------------------------------------------

def plot_crisis_calm_contrast(
    snapshots: dict[str, dict[str, dict[str, Any]]],
    cot_mats: dict[str, np.ndarray],
    return_mats: dict[str, np.ndarray],
    out_dir: Path | None = None,
) -> Path | None:
    """
    2×2 figure: best-signal crisis week (row 1) vs worst-signal calm week (row 2).
    Always attempts to save even if individual panels fail.
    """
    out_dir = out_dir or _DENDROGRAMS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    crisis_peaks = {k: snapshots[k]["peak"] for k in CRISIS_PERIODS if k in snapshots}
    calm_peaks   = {k: snapshots[k]["peak"] for k in CALM_PERIODS   if k in snapshots}

    if not crisis_peaks or not calm_peaks:
        warnings.warn("Cannot build contrast figure — missing crisis or calm peaks.")
        return None

    best_crisis  = max(crisis_peaks, key=lambda k: crisis_peaks[k]["ARI_h2"])
    worst_calm   = min(calm_peaks,   key=lambda k: calm_peaks[k]["ARI_h2"])
    crisis_date  = crisis_peaks[best_crisis]["date"]
    calm_date    = calm_peaks[worst_calm]["date"]
    crisis_ari   = crisis_peaks[best_crisis]["ARI_h2"]
    calm_ari     = calm_peaks[worst_calm]["ARI_h2"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "COT vs Return Co-movement Dendrograms: Crisis (top) vs Calm (bottom)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    row_meta = [
        (best_crisis, crisis_date, crisis_ari),
        (worst_calm,  calm_date,   calm_ari),
    ]
    panel_specs = [
        (cot_mats,    "COT features"),
        (return_mats, "Return co-movement"),
    ]

    for row, (period, date_str, ari_val) in enumerate(row_meta):
        # Row label
        label = (f"{'Crisis' if row == 0 else 'Calm'}: "
                 f"{period.replace('_', ' ')}\n{date_str}  |  ARI={ari_val:.3f}")
        axes[row, 0].annotate(
            label, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-110, 0), textcoords="offset points",
            ha="right", va="center", fontsize=8, fontweight="bold", rotation=90,
        )
        for col, (mats, panel_title) in enumerate(panel_specs):
            ax = axes[row, col]
            try:
                if date_str in mats:
                    _draw_panel(ax, mats[date_str], f"{panel_title}\n{date_str}")
                else:
                    ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")
                    ax.set_title(f"{panel_title}\n{date_str}", fontsize=9)
            except Exception as exc:
                ax.text(0.5, 0.5, f"Error: {exc}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="red")

    fig.legend(handles=_sector_legend(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=8, frameon=False)
    plt.tight_layout()

    path = out_dir / "dendrogram_crisis_calm_contrast.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path
