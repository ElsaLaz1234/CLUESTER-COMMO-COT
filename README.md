# Do COT Positioning Clusters Anticipate Commodity Return Co-movement?

A cross-sectional clustering study that tests whether CFTC Commitment of Traders (COT) institutional positioning clusters predict commodity futures return co-movement structure at 1–3 month horizons. The project introduces realized return clustering as the target variable — a structurally different hypothesis from individual return prediction — and applies hierarchical clustering with Ward linkage to both COT positioning and realized returns across 9 commodity futures from 2006 to 2024.

---

## Motivation

The CFTC Commitment of Traders report is one of the few publicly available sources of institutional positioning data with weekly frequency and clean trader category decomposition. Commodity futures markets are structurally bipartite — commercial hedgers hedge physical exposure while managed money (hedge funds, CTAs) speculate on price direction. This separation gives COT a direct economic interpretation.

The central question is not whether COT predicts individual commodity returns — Wang & Zhang (2023) establish this at monthly frequency using machine learning — but whether the cross-commodity co-movement structure embedded in institutional positioning anticipates the cross-commodity co-movement structure of realized returns. If managed money have information about a common fundamental driver across a group of commodities — an energy supply shock, an agricultural drought, a dollar move — they should position coordinately on that group before prices reflect this information. The co-movement structure of their positioning should therefore precede the co-movement structure of realized returns.

---

## Objectives

This project tests the following hypothesis :

*Do clusters formed from COT positioning at t−h predict clusters formed from realized returns at t, and at what horizon?*

Three sub-questions structure the analysis :

**Signal existence** — is the aggregate ARI between COT clusters and return clusters statistically significantly above zero after correcting for serial autocorrelation?

**Optimal horizon** — at which lag h does the predictive signal peak, and is this consistent with typical managed money holding periods?

**Signal heterogeneity** — does the signal vary across market regimes (gradual cycles vs acute crashes vs calm periods), COT trader categories, and return cluster stability regimes?

---

## Data and Method

**Universe** : 9 commodity futures, front-month, 2006–2024. Source : Yahoo Finance.

| Sector | Tickers |
|--------|---------|
| Energy | CL (WTI Crude), NG (Natural Gas), RB (RBOB Gasoline) |
| Metals | GC (Gold), SI (Silver), HG (Copper) |
| Agri | ZC (Corn), ZW (Wheat), ZS (Soybean) |

**COT data** : CFTC Disaggregated Report, weekly, 2006–2024. Three features extracted from the Managed Money category per commodity per week :
- `MM_net` : Managed Money Longs − Managed Money Shorts
- `ΔMM_net` : week-over-week change in net positioning
- `COM/MM ratio` : Commercial Net / Managed Money Net

All features normalized via z-score on a 52-week rolling window to avoid data leakage.

**Two clustering objects** are constructed each week :

*COT positioning clusters* — hierarchical clustering (Ward linkage, correlation distance D = 1−ρ, k=3) on the normalized COT feature matrix (9 commodities × 3 features).

*Return co-movement clusters* — hierarchical clustering on the past 16 weeks of raw realized log returns (9 commodities × 16 weekly returns). Raw returns are used deliberately — two commodities in the same return cluster have empirically moved together over the past 4 months, which is an observable fact rather than a feature engineering choice.

**Lead-lag test** : for each week t and lag h ∈ {0, 4, 8, 12} weeks (corresponding to 0, 1, 2, 3 months) :

```
ARI_h(t) = ARI(C_{t-h}^{COT}, C_t^{ret})
lift_h   = mean(ARI_h) - mean(ARI_0)
```

**Significance testing** : Newey-West HAC corrected t-statistics to account for serial autocorrelation introduced by the 52-week rolling window.

**Sensitivity analyses** :
- Five COT trader category specifications (M1_MM through M5_combined)
- Subperiod decomposition across 4 crisis and 3 calm periods
- Cluster stability conditioning (stable vs transition weeks)
- Weekly robustness check (26-week window, h = 1,2,3,4 weeks)

---

## Results

**The signal exists and is statistically significant.** All four lags show positive mean ARI after Newey-West HAC correction (t-statistics 5.32–5.79, all p < 0.001). The 2-month horizon is the optimal and most robust predictive window (mean ARI = 0.049, t = 5.79), with lift turning positive at h=2m (+0.006) after a negative dip at h=1m (−0.002) — consistent with typical managed money holding periods of 6–12 weeks.

**The signal is regime-dependent and episodic.** The aggregate result masks substantial heterogeneity. During gradual commodity cycles (Oil crash 2014-2015, Gas shock 2021-2022), ARI increases monotonically with lag — reaching 0.073 at h=3m during the oil crash — confirming genuine anticipatory positioning where managed money build sustained cross-sectoral views months before co-movement materialises in prices. During acute crashes (GFC 2008, COVID 2020), the signal inverts at longer lags as forced liquidation renders prior positioning irrelevant. During calm periods the signal peaks at 1 month rather than 2, consistent with faster transmission of quieter macro themes.

**Combining all COT categories substantially outperforms Managed Money alone.** M5_combined (all trader categories concatenated) produces ARI of 0.073–0.075 across all lags, versus 0.041–0.049 for M1_MM. Predictive information about cross-commodity co-movement is distributed across the full institutional positioning landscape — commercial hedgers, swap dealers, and other reportables all contribute complementary information.

**Within-subperiod variance reveals the episodic nature of the signal.** The same subperiod contains peak weeks with ARI up to 0.739 and trough weeks as low as −0.246 at h=2m. Peak weeks correspond to moments where the macro framework embedded in COT positioning at t−2m remained valid and undisrupted at t. Trough weeks correspond to moments where structural breaks, forced liquidation, or regime transitions invalidated prior positioning before it could materialise — through two distinct mechanisms : forced liquidation during acute crashes, and regime transition timing during gradual shock onsets where an OPEC announcement or supply disruption reconfigured co-movement faster than the 2-month lag could capture.

**An open momentum question.** The consistently positive synchronous ARI (h=0) across all subperiods — including crisis periods where the lagged signal breaks down entirely — raises an intriguing question. COT positioning may be better understood as a momentum indicator reflecting active institutional flows currently driving commodity co-movement, rather than as a predictor of future structure. If managed money trend-following behavior generates contemporaneous co-movement through progressive price impact, the COT cluster signal at h=0 would be capturing that momentum in real time rather than anticipating what comes next. Whether this interpretation translates into a viable weekly rebalancing signal remains an open empirical question for future work.

---

## Limitations

- **Small universe** : 9 assets with k=3 constrains the number of possible partitions and limits the statistical power of ARI-based tests. Extension to 20+ CFTC commodities is the primary v2 target.
- **ARI is a discrete metric** : the Mantel test on continuous distance matrices is more powerful and does not require a k choice — planned for v2.
- **No formal subperiod significance testing** : Newey-West correction is applied to the full-sample ARI series only. Subperiod samples (18–36 weeks for crisis periods) are too small for reliable inference.
- **COT publication lag** : the COT report is published Friday with data as of the preceding Tuesday — a 3-day lag that is negligible at weekly frequency but is documented for completeness.
- **Single market, single asset class** : results reflect commodity futures only and may not generalise to other asset classes or non-US markets.
- **No trading implementation** : ARI and lift measure cluster agreement between COT positioning and return co-movement structures — they are statistical signal quality metrics, not performance metrics. No portfolio construction, position sizing, or return backtest is performed in this version. The question of whether the signal translates into a profitable trading strategy is left for the v2 momentum reinterpretation extension.

---

## References

Wang, S., & Zhang, T. (2023). Predictability of Commodity Futures Returns with Machine Learning Models. *Journal of Futures Markets*, 44(2), 302–322. https://doi.org/10.1002/fut.22471

De Roon, F. A., Nijman, T. E., & Veld, C. H. (2000). Hedging Pressure Effects in Futures Markets. *The Journal of Finance*, 55(3), 1437–1456. https://doi.org/10.1111/0022-1082.00253

Bessembinder, H. (1992). Systematic Risk, Hedging Pressure, and Risk Premiums in Futures Markets. *Review of Financial Studies*, 5(4), 637–667.

---

## User Guide

```
COMMO_COT/
│
├── config.py                    # Central configuration — rolling window, lags,
│                                #   subperiods, universe, clustering parameters
│
├── main.py                      # Entry point / CLI
│                                #   --mode csv         → full pipeline from CSV data
│                                #   --mode category    → COT category decomposition
│                                #   --mode subperiod   → subperiod lead-lag analysis
│                                #   --mode dendrograms → snapshot dendrogram figures
│                                #   --mode significance→ Newey-West t-statistics
│                                #   --mode option_b    → return co-movement clustering
│
├── data/
│   ├── data_loader.py           # COT and futures price ingestion
│   └── data_cot.py              # CFTC Disaggregated COT parsing
│
├── src/
│   ├── clustering.py            # Hierarchical clustering (Ward, correlation distance)
│   ├── features.py              # COT feature construction and z-score normalization
│   ├── return_clustering.py     # Return co-movement cluster construction
│   ├── return_lead_lag.py       # ARI/NMI lead-lag computation
│   ├── significance_testing.py  # Newey-West HAC t-statistics
│   ├── subperiod_analysis.py    # Subperiod and regime decomposition
│   ├── cot_category_features.py # M1–M5 category feature matrices
│   └── dendrograms.py           # Snapshot dendrogram visualization
│
└── results/                     # All outputs — figures, CSVs, parquet files
```

**Run order** :

```bash
python main.py --mode csv            # Build COT and price clusters
python main.py --mode option_b       # Build return co-movement clusters and lead-lag
python main.py --mode category       # COT category decomposition
python main.py --mode subperiod      # Subperiod analysis
python main.py --mode dendrograms    # Snapshot dendrograms
python main.py --mode significance   # Newey-West significance testing
```
