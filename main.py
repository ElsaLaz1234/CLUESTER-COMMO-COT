"""
COMMO_COT — entry point.

Usage
-----
python main.py                          # full pipeline
python main.py --refresh                # re-download all data
python main.py --no-plots               # skip plot generation
python main.py --mode dendrograms       # dendrogram snapshots only
python main.py --mode significance      # Newey-West HAC significance testing
"""
import argparse
import sys
from pathlib import Path

from config import RESULTS_DIR, PERIOD_SLUG


def _print_config() -> None:
    from config import ROLLING_WINDOW, RETURN_LAGS, LAGS, SNAPSHOT_LAG, LAG_LABELS
    print("Config:")
    print(f"  ROLLING_WINDOW = {ROLLING_WINDOW}w (12 months)")
    print(f"  RETURN_LAGS    = {RETURN_LAGS}w (4 months)")
    print(f"  LAGS           = {LAGS} weeks → {[LAG_LABELS[l] for l in LAGS]}")
    print(f"  SNAPSHOT_LAG   = {SNAPSHOT_LAG}w ({LAG_LABELS[SNAPSHOT_LAG]})")
    print()


def run_pipeline(refresh: bool, make_plots: bool) -> None:
    _print_config()
    from pipeline import run
    run(refresh=refresh, make_plots=make_plots)


def run_dendrograms() -> None:
    import pandas as pd
    from pipeline import load_data
    from src.dendrograms import (
        identify_snapshot_weeks,
        build_feature_matrices,
        plot_snapshot_dendrograms,
        plot_crisis_calm_contrast,
    )

    lead_lag_csv = Path(RESULTS_DIR) / "lead_lag_MM_full.csv"
    if not lead_lag_csv.exists():
        print(f"ERROR: {lead_lag_csv} not found.")
        print("Run `python main.py` first to generate pipeline outputs.")
        sys.exit(1)

    print(f"Loading ARI data from {lead_lag_csv}...")
    ari_df = pd.read_csv(lead_lag_csv)

    _, log_ret, cot, dates = load_data()
    cot_mats, return_mats = build_feature_matrices(cot, log_ret, dates)

    snapshots = identify_snapshot_weeks(ari_df)

    print("Generating per-period dendrogram figures...")
    saved = plot_snapshot_dendrograms(snapshots, cot_mats, return_mats)

    print("Generating crisis-vs-calm contrast figure...")
    contrast = plot_crisis_calm_contrast(snapshots, cot_mats, return_mats)

    _print_checklist(saved, contrast)
    print("Dendrogram snapshots complete.")


def _print_checklist(saved: list[Path], contrast: "Path | None") -> None:
    from config import CRISIS_PERIODS, CALM_PERIODS
    expected = [
        f"dendrogram_{PERIOD_SLUG[p]}_{k}.png"
        for p in list(CRISIS_PERIODS) + list(CALM_PERIODS)
        for k in ("peak", "trough")
    ] + ["dendrogram_crisis_calm_contrast.png"]

    out_dir = Path(RESULTS_DIR) / "dendrograms"
    print("\n" + "=" * 52)
    print("OUTPUT CHECKLIST")
    print("=" * 52)
    all_ok = True
    for fname in expected:
        exists = (out_dir / fname).exists()
        print(f"  [{'OK ' if exists else 'MISSING'}]  {fname}")
        if not exists:
            all_ok = False
    print("=" * 52)
    if not all_ok:
        print("  Some files missing — see warnings above.")
    print()


def run_significance() -> None:
    from src.significance_testing import compute_significance

    lead_lag_csv = Path(RESULTS_DIR) / "lead_lag_MM_full.csv"
    if not lead_lag_csv.exists():
        print("ERROR: results/lead_lag_MM_full.csv not found.")
        print("Run main pipeline first: python main.py")
        sys.exit(1)

    compute_significance(lead_lag_csv)
    print(f"  results/significance_summary.csv")
    print(f"  results/significance_summary.png")
    print("Significance testing complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="COMMO_COT pipeline")
    parser.add_argument("--mode", choices=["pipeline", "dendrograms", "significance"],
                        default="pipeline")
    parser.add_argument("--refresh",  action="store_true",
                        help="Re-download raw data")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation (pipeline mode only)")
    args = parser.parse_args()

    if args.mode == "dendrograms":
        run_dendrograms()
    elif args.mode == "significance":
        run_significance()
    else:
        run_pipeline(refresh=args.refresh, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
