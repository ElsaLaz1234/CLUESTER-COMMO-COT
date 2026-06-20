# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

TICKERS = {
    "energy": ["CL=F", "NG=F", "RB=F"],
    "metals": ["GC=F", "SI=F", "HG=F"],
    "agri":   ["ZC=F", "ZW=F", "ZS=F"],
}
ALL_TICKERS   = [t for group in TICKERS.values() for t in group]
TICKERS_SHORT = [t.replace("=F", "") for t in ALL_TICKERS]  # single source of truth

TICKER_LABELS = {
    "CL=F": "CRUDE OIL",   "NG=F": "NATURAL GAS",    "RB=F": "RBOB GASOLINE",
    "GC=F": "GOLD",        "SI=F": "SILVER",          "HG=F": "COPPER",
    "ZC=F": "CORN",        "ZW=F": "WHEAT",           "ZS=F": "SOYBEANS",
}

# Sector color coding used in dendrogram figures
SECTOR_COLORS: dict[str, str] = {
    "CL": "red",        "NG": "red",        "RB": "red",
    "GC": "darkorange", "SI": "darkorange", "HG": "darkorange",
    "ZC": "green",      "ZW": "green",      "ZS": "green",
}

# ---------------------------------------------------------------------------
# Time range and clustering parameters
# ---------------------------------------------------------------------------

START_DATE = "2006-01-01"
END_DATE   = "2024-12-31"

ROLLING_WINDOW       = 52   # weeks — z-score normalisation window (≈ 12 months)
RETURN_LAGS          = 16   # weeks of raw log-return history for return clusters (≈ 4 months)
N_CLUSTERS           = 3
WEEKS_PER_MONTH      = 4    # conversion factor used for lag expression
LAGS                 = [0, 4, 8, 12]   # weeks: 0, 1 month, 2 months, 3 months
SNAPSHOT_LAG         = 8    # weeks — lag used for dendrogram peak/trough selection (2 months)

# Display labels for lag values — used in plot titles, axis labels, print statements
LAG_LABELS: dict[int, str] = {0: "h=0 (sync)", 4: "h=1m", 8: "h=2m", 12: "h=3m"}
COPHENETIC_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Sub-periods
# ---------------------------------------------------------------------------

CRISIS_PERIODS: list[str] = [
    "GFC_2008_2009",
    "Oil_crash_2014_2015",
    "COVID_2020",
    "Gas_shock_2021_2022",
]
CALM_PERIODS: list[str] = [
    "Baseline_2016_2017",
    "Baseline_2018_2019",
    "Baseline_2023",
]
SUBPERIODS: dict[str, tuple[str, str]] = {
    "GFC_2008_2009":       ("2008-01-01", "2009-12-31"),
    "Oil_crash_2014_2015": ("2014-07-01", "2016-01-01"),
    "COVID_2020":          ("2020-01-01", "2020-12-31"),
    "Gas_shock_2021_2022": ("2021-01-01", "2022-12-31"),
    "Baseline_2016_2017":  ("2016-02-01", "2017-12-31"),
    "Baseline_2018_2019":  ("2018-01-01", "2019-12-31"),
    "Baseline_2023":       ("2023-01-01", "2023-12-31"),
}

# Short slugs used in output filenames
PERIOD_SLUG: dict[str, str] = {
    "GFC_2008_2009":       "gfc_2008",
    "Oil_crash_2014_2015": "oil_crash_2014",
    "COVID_2020":          "covid_2020",
    "Gas_shock_2021_2022": "gas_shock_2022",
    "Baseline_2016_2017":  "baseline_2016",
    "Baseline_2018_2019":  "baseline_2018",
    "Baseline_2023":       "baseline_2023",
}

# ---------------------------------------------------------------------------
# COT categories
# ---------------------------------------------------------------------------

COT_CATEGORIES = {
    "MM": "Managed Money",
    "PM": "Producer/Merchant/Processor/User",
    "SD": "Swap Dealers",
    "OR": "Other Reportables",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR    = "data"
RESULTS_DIR = "results"
