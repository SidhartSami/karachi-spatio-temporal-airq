"""
Phase 3 — Step 3: Merge Ground-Truth PM2.5 into Feature Dataset
================================================================
What this script does:
  1. Loads the clean feature dataset (output of Step 1).
  2. Loads the OpenAQ ground-truth PM2.5 CSV (output of Step 2).
  3. Performs a smart merge:
       a. Exact station-level match where available.
       b. Falls back to the US Consulate city-wide anchor for stations
          with no direct OpenAQ sensor coverage.
  4. Reports coverage statistics per station.
  5. Drops rows with no PM2.5 target (can't train on them).
  6. Saves the final modeling-ready dataset:
       data/processed/modeling_dataset.csv

This is the dataset that feeds Phase 4 (ML Modeling) directly.

Usage:
  python phase3_step3_merge_target.py \
      --features  data/processed/merged_clean.csv \
      --groundtruth data/raw/openaq_pm25_karachi.csv \
      --output    data/processed/modeling_dataset.csv

Dependencies:
  pip install pandas numpy
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Name of the US Consulate anchor row in the ground truth CSV
US_CONSULATE_STATION = "US_Consulate_Karachi"

# Name mapping: your GEE station names → OpenAQ station names fetched in Step 2
# Adjust if OpenAQ returned different name strings.
STATION_NAME_MAP = {
    "Gulshan-e-Iqbal":   "Gulshan-e-Iqbal",
    "Gulshan_e_Iqbal":   "Gulshan-e-Iqbal",
    "Saddar":            "Saddar",
    "SITE_Industrial":   "SITE_Industrial",
    "Korangi_Industrial":"Korangi_Industrial",
    "North_Nazimabad":   "North_Nazimabad",
    "Gulistan_Jauhar":   "Gulistan_Jauhar",
    "Landhi":            "Landhi",
    "Federal_B_Area":    "Federal_B_Area",
}


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")   # normalise to string key
    log.info("Features  shape=%s", df.shape)
    return df


def load_groundtruth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = df["date"].astype(str)
    log.info("Ground truth  shape=%s  stations=%s",
             df.shape, df["station"].unique().tolist())
    return df


# ── Merge Logic ───────────────────────────────────────────────────────────────

def build_pm25_lookup(gt: pd.DataFrame) -> tuple[dict, pd.Series]:
    """
    Returns:
      station_daily: dict  {station_name → {date → pm25_mean}}
      anchor_series: Series  {date → pm25_mean}  (US Consulate)
    """
    station_daily: dict = {}
    anchor_series = pd.Series(dtype=float)

    for station, grp in gt.groupby("station"):
        series = grp.set_index("date")["pm25_mean"]
        if station == US_CONSULATE_STATION:
            anchor_series = series
            log.info("Anchor (US Consulate): %d daily records", len(series))
        else:
            station_daily[station] = series.to_dict()

    return station_daily, anchor_series


def merge_target(
    features: pd.DataFrame,
    gt: pd.DataFrame,
) -> pd.DataFrame:
    station_daily, anchor_series = build_pm25_lookup(gt)

    pm25_values   = []
    pm25_sources  = []   # "exact", "anchor", or "none"

    for _, row in features.iterrows():
        date        = row["date"]
        station_raw = row["station"]
        station     = STATION_NAME_MAP.get(station_raw, station_raw)

        # 1. Try exact station match
        if station in station_daily and date in station_daily[station]:
            val = station_daily[station][date]
            pm25_values.append(val)
            pm25_sources.append("exact")

        # 2. Fall back to US Consulate anchor
        elif date in anchor_series.index:
            pm25_values.append(anchor_series[date])
            pm25_sources.append("anchor")

        # 3. No coverage at all
        else:
            pm25_values.append(np.nan)
            pm25_sources.append("none")

    features = features.copy()
    features["pm25"]        = pm25_values
    features["pm25_source"] = pm25_sources
    return features


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_coverage(df: pd.DataFrame) -> None:
    log.info("─" * 60)
    log.info("PM2.5 target coverage by station:")
    summary = (
        df.groupby("station")["pm25_source"]
        .value_counts()
        .unstack(fill_value=0)
        .assign(
            total=lambda x: x.sum(axis=1),
            pct_exact=lambda x: (x.get("exact", 0) / x["total"] * 100).round(1),
            pct_anchor=lambda x: (x.get("anchor", 0) / x["total"] * 100).round(1),
            pct_none=lambda x: (x.get("none", 0) / x["total"] * 100).round(1),
        )
    )
    for station, row in summary.iterrows():
        log.info(
            "  %-25s  exact=%5.1f%%  anchor=%5.1f%%  missing=%5.1f%%",
            station, row["pct_exact"], row["pct_anchor"], row["pct_none"]
        )
    log.info("─" * 60)


# ── Final Cleaning & Validation ───────────────────────────────────────────────

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["pm25"])
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d rows with no PM2.5 target (no coverage).", dropped)

    # Sanity: PM2.5 physical range check (0–1000 µg/m³)
    invalid = (df["pm25"] < 0) | (df["pm25"] > 1000)
    if invalid.any():
        log.warning("Removing %d rows with PM2.5 outside [0, 1000].", invalid.sum())
        df = df[~invalid]

    # Reorder: date, station, pm25 first, then features
    priority_cols = ["date", "station", "pm25", "pm25_source"]
    other_cols    = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    log.info("Final dataset: %d rows × %d columns", *df.shape)
    log.info("PM2.5 stats:  mean=%.1f  std=%.1f  min=%.1f  max=%.1f",
             df["pm25"].mean(), df["pm25"].std(),
             df["pm25"].min(), df["pm25"].max())
    return df.reset_index(drop=True)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("✅ Modeling dataset saved → %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",     required=True,
                        help="Path to Step 1 output (merged_clean.csv)")
    parser.add_argument("--groundtruth",  required=True,
                        help="Path to Step 2 output (openaq_pm25_karachi.csv)")
    parser.add_argument("--output",       required=True,
                        help="Where to save modeling_dataset.csv")
    args = parser.parse_args()

    features_path = Path(args.features)
    gt_path       = Path(args.groundtruth)
    output_path   = Path(args.output)

    for p in [features_path, gt_path]:
        if not p.exists():
            log.error("File not found: %s", p)
            sys.exit(1)

    features = load_features(features_path)
    gt       = load_groundtruth(gt_path)
    merged   = merge_target(features, gt)
    report_coverage(merged)
    final    = finalize(merged)
    save(final, output_path)

    log.info("")
    log.info("Next step → Phase 4: Machine Learning Modeling")
    log.info("  Input file : %s", output_path)
    log.info("  Target col : pm25")
    log.info("  Feature cols (drop before training): date, station, pm25_source")


if __name__ == "__main__":
    main()
