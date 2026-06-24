"""
CFTC Disaggregated COT Report — download, parse, and cache.
"""
import io
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import TICKERS_SHORT, DATA_DIR

Path(DATA_DIR).mkdir(exist_ok=True)

_CFTC_URLS = {
    "hist": "https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_2006_2016.zip",
    **{yr: f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{yr}.zip"
       for yr in range(2017, 2025)},
}

_COMMODITY_KEYWORDS = {
    "CL": ["CRUDE OIL, LIGHT SWEET", "WTI-PHYSICAL", "CRUDE OIL"],
    "NG": ["NATURAL GAS"],
    "RB": ["RBOB GASOLINE", "GASOLINE"],
    "GC": ["GOLD"],
    "SI": ["SILVER"],
    "HG": ["COPPER-GRADE #1", "COPPER"],
    "ZC": ["CORN"],
    "ZW": ["WHEAT", "CHICAGO WHEAT"],
    "ZS": ["SOYBEANS"],
}


def _download_cot_chunk(key) -> pd.DataFrame:
    cache = Path(DATA_DIR) / f"cot_{key}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = _CFTC_URLS[key]
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        fname = [n for n in z.namelist() if n.endswith(".txt")][0]
        with z.open(fname) as f:
            df = pd.read_csv(f, low_memory=False)

    df.to_parquet(cache)
    return df


def _parse_cot_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() > 0.5:
        return parsed
    try:
        parsed = pd.to_datetime(series.astype(str).str.zfill(6), format="%y%m%d", errors="coerce")
        if parsed.notna().mean() > 0.5:
            return parsed
    except Exception:
        pass
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


def _match_commodity(name: str, ticker: str) -> bool:
    name_up = name.upper()
    return any(kw in name_up for kw in _COMMODITY_KEYWORDS.get(ticker, []))


def load_cot(refresh: bool = False) -> pd.DataFrame:
    """Return weekly COT disaggregated data for our 9 commodities."""
    cache = Path(DATA_DIR) / "cot_processed.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    frames = []
    for key in list(_CFTC_URLS.keys()):
        try:
            frames.append(_download_cot_chunk(key))
            print(f"  COT chunk '{key}' OK")
        except Exception as e:
            warnings.warn(f"COT chunk '{key}' failed: {e}")

    if not frames:
        raise RuntimeError("No COT data downloaded.")

    raw = pd.concat(frames, ignore_index=True)

    date_col_candidates = [
        "Report_Date_as_YYYY-MM-DD",
        "As_of_Date_In_Form_YYMMDD",
        "Report_Date_as_MM_DD_YYYY",
    ] + [c for c in raw.columns if "date" in c.lower() or "as_of" in c.lower()]

    date_col = next((c for c in date_col_candidates if c in raw.columns), None)
    if date_col is None:
        raise RuntimeError(f"Cannot find date column. Columns: {list(raw.columns[:20])}")

    raw["date"] = _parse_cot_date(raw[date_col])
    bad_dates = raw["date"].isna().sum()
    if bad_dates > 0:
        warnings.warn(f"{bad_dates} rows dropped due to unparseable dates")
    raw = raw.dropna(subset=["date"])
    raw = raw[raw["date"].dt.year >= 2006]

    print(f"  COT raw rows after date filter: {len(raw):,}  "
          f"({raw['date'].min().date()} – {raw['date'].max().date()})")

    col_map = {
        "M_Money_Positions_Long_All":    ("MM", "longs"),
        "M_Money_Positions_Short_All":   ("MM", "shorts"),
        "Prod_Merc_Positions_Long_All":  ("PM", "longs"),
        "Prod_Merc_Positions_Short_All": ("PM", "shorts"),
        "Swap_Positions_Long_All":       ("SD", "longs"),
        "Swap__Positions_Short_All":     ("SD", "shorts"),
        "Other_Rept_Positions_Long_All": ("OR", "longs"),
        "Other_Rept_Positions_Short_All":("OR", "shorts"),
    }
    col_map = {k: v for k, v in col_map.items() if k in raw.columns}
    if not col_map:
        raise RuntimeError(
            f"No COT position columns matched. Available: {[c for c in raw.columns if 'Long' in c or 'Short' in c]}"
        )

    name_col = next(
        (c for c in ("Market_and_Exchange_Names", "Commodity_Name") if c in raw.columns),
        raw.columns[0],
    )

    for col in col_map:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    records = []
    for ticker in TICKERS_SHORT:
        mask = raw[name_col].astype(str).apply(lambda x: _match_commodity(x, ticker))
        sub = raw[mask].copy()
        if sub.empty:
            warnings.warn(f"No COT rows matched for {ticker}")
            continue
        sub = sub.sort_values("date").drop_duplicates("date", keep="last")

        for _, row in sub.iterrows():
            date = row["date"]
            for cat in ["MM", "PM", "SD", "OR"]:
                long_col  = next((c for c, (ct, s) in col_map.items() if ct == cat and s == "longs"),  None)
                short_col = next((c for c, (ct, s) in col_map.items() if ct == cat and s == "shorts"), None)
                if long_col is None or short_col is None:
                    continue
                longs  = row[long_col]
                shorts = row[short_col]
                net    = (longs - shorts) if pd.notna(longs) and pd.notna(shorts) else np.nan
                records.append({
                    "date": date, "ticker": ticker, "category": cat,
                    "longs": longs, "shorts": shorts, "net": net,
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("COT DataFrame is empty after parsing.")

    df = df.sort_values(["ticker", "category", "date"]).reset_index(drop=True)
    df["delta_net"] = df.groupby(["ticker", "category"])["net"].diff()

    mm = df[df["category"] == "MM"][["date", "ticker", "net"]].rename(columns={"net": "mm_net"})
    pm = df[df["category"] == "PM"][["date", "ticker", "net"]].rename(columns={"net": "pm_net"})
    ratio_df = mm.merge(pm, on=["date", "ticker"])
    ratio_df["com_mm_ratio"] = ratio_df["pm_net"] / ratio_df["mm_net"].replace(0, np.nan)
    df = df.merge(ratio_df[["date", "ticker", "com_mm_ratio"]], on=["date", "ticker"], how="left")

    df.to_parquet(cache)
    return df
