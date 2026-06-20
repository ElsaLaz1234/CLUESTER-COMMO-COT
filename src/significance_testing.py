"""
Newey-West HAC significance testing for COT → return co-movement lead-lag signal.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

# Project root is one level up
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LAGS, LAG_LABELS, RESULTS_DIR

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})


def compute_significance(ari_csv_path: str | Path) -> pd.DataFrame:
    """
    Test whether mean ARI at each lag is significantly different from zero,
    using Newey-West HAC-corrected standard errors.

    Null hypothesis
    ---------------
    H0: E[ARI_h(t)] = 0  for each lag h.
    A rejection means COT clusters at t-h contain statistically significant
    information about return co-movement clusters at t.

    Newey-West correction rationale
    --------------------------------
    ARI_h(t) is serially autocorrelated by construction: the 52-week rolling
    window used to build COT features overlaps heavily across consecutive weeks
    (51/52 weeks in common). Return features share 15/16 weekly observations
    across consecutive weeks. OLS standard errors assuming i.i.d. errors would
    severely understate uncertainty. Newey-West HAC standard errors are
    consistent under arbitrary heteroskedasticity and autocorrelation up to
    the chosen bandwidth.

    Bandwidth selection
    -------------------
    Automatic bandwidth: nlags = int(4 * (T/100)^(2/9)), following the
    rule-of-thumb in Andrews (1991) as implemented in statsmodels.

    Interpretation
    --------------
    A significant positive mean ARI (t > 0, p < 0.10) at h > 0 indicates
    that COT positioning clusters anticipate return co-movement clusters.
    Significance at h=0 (synchronous) serves as a benchmark — it measures
    contemporaneous alignment, not prediction.

    Parameters
    ----------
    ari_csv_path : path to lead_lag_MM_full.csv

    Returns
    -------
    DataFrame with columns:
        lag_label, lag_weeks, mean_ARI, NW_SE, t_stat, p_value,
        significant_10pct, significant_5pct
    """
    df = pd.read_csv(ari_csv_path)
    df["date"] = pd.to_datetime(df["date"])

    results = []
    bandwidths = []

    for lag_weeks in LAGS:
        series = (
            df[df["lag"] == lag_weeks]["ari"]
            .dropna()
            .values
        )
        T = len(series)

        # OLS of ARI on a constant → tests H0: mean = 0
        model = sm.OLS(series, np.ones(T)).fit()

        # Newey-West bandwidth: Andrews (1991) rule-of-thumb
        nlags = int(4 * (T / 100) ** (2 / 9))
        bandwidths.append(nlags)

        nw_cov = cov_hac(model, nlags=nlags)
        nw_se  = float(np.sqrt(nw_cov[0, 0]))
        t_stat = float(model.params[0] / nw_se)

        # One-sided p-value: H1 is mean ARI > 0
        p_value = float(1 - scipy.stats.t.cdf(t_stat, df=T - 1))

        results.append({
            "lag_label":        LAG_LABELS.get(lag_weeks, f"h={lag_weeks}"),
            "lag_weeks":        lag_weeks,
            "mean_ARI":         float(model.params[0]),
            "NW_SE":            nw_se,
            "t_stat":           t_stat,
            "p_value":          p_value,
            "significant_10pct": p_value < 0.10,
            "significant_5pct":  p_value < 0.05,
        })

    summary = pd.DataFrame(results)
    nw_bw   = int(np.mean(bandwidths))  # same T across lags → single value

    _print_summary(summary, nw_bw)
    _save_csv(summary)
    _save_figure(summary)

    return summary


# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame, nw_bw: int) -> None:
    print()
    print("=" * 54)
    print("SIGNIFICANCE TESTING — Newey-West HAC corrected")
    print("=" * 54)
    print(f"{'Lag':<12} {'Mean_ARI':>9} {'NW_SE':>7} {'t_stat':>7} {'p_value':>8}  Sig")
    print("-" * 54)
    for _, row in df.iterrows():
        sig = ("**" if row["significant_5pct"]
               else "*" if row["significant_10pct"]
               else "")
        print(
            f"{row['lag_label']:<12} "
            f"{row['mean_ARI']:>9.4f} "
            f"{row['NW_SE']:>7.4f} "
            f"{row['t_stat']:>7.3f} "
            f"{row['p_value']:>8.4f}  {sig}"
        )
    print("=" * 54)
    print("* p<0.10   ** p<0.05   (one-sided, H1: mean ARI > 0)")
    print(f"NW bandwidth used: {nw_bw} lags")
    print("=" * 54)
    print()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def _save_csv(df: pd.DataFrame) -> Path:
    out = Path(RESULTS_DIR) / "significance_summary.csv"
    df.to_csv(out, index=False)
    print(f"  Saved {out}")
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _save_figure(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    x      = np.arange(len(df))
    labels = df["lag_label"].tolist()
    means  = df["mean_ARI"].values
    ci     = 1.96 * df["NW_SE"].values

    colors = [
        "mediumseagreen" if r["significant_5pct"]
        else "darkorange"  if r["significant_10pct"]
        else "tomato"
        for _, r in df.iterrows()
    ]

    ax.bar(x, means, color=colors, alpha=0.85, width=0.5,
           yerr=ci, capsize=5, error_kw={"elinewidth": 1.2, "ecolor": "#333"})

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Lag", labelpad=8)
    ax.set_ylabel("Mean ARI")
    ax.set_title("Mean ARI with Newey-West 95% confidence intervals (HAC corrected)",
                 pad=10)

    # Annotate inside each bar (midpoint), white text for contrast
    for i, (_, row) in enumerate(df.iterrows()):
        bar_mid = means[i] / 2
        ax.text(
            i, bar_mid,
            f"t = {row['t_stat']:.2f}\np = {row['p_value']:.3f}",
            ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold",
        )

    # Legend below the plot — never overlaps bars
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color="mediumseagreen", label="p < 0.05"),
        Patch(color="darkorange",     label="p < 0.10"),
        Patch(color="tomato",         label="not significant"),
    ]
    ax.legend(handles=legend_patches, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = Path(RESULTS_DIR) / "significance_summary.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")
    return out
