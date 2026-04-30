"""
Phase 3 — Step 1: Clean Dead Columns & Impute AOD Missing Values
================================================================
What this script does:
  1. Drops 100%-missing columns: no2, so2, co
  2. Imputes ~10% missing in Optical_Depth_047 & Optical_Depth_055
     using KNN Imputation, stratified by (station, month) for spatial
     and seasonal coherence.
  3. Saves a clean intermediate CSV: data/processed/merged_clean.csv

Usage:
  python phase3_step1_clean_impute.py \
      --input  data/processed/merged_karachi_dataset.csv \
      --output data/processed/merged_clean.csv

Dependencies:
  pip install pandas numpy scikit-learn
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEAD_COLUMNS = ["no2", "so2", "co"]          # 100 % missing — drop entirely
AOD_COLUMNS  = ["Optical_Depth_047", "Optical_Depth_055"]  # ~10 % — impute

# Features used as context when doing KNN imputation.
# These are fully observed numeric columns that carry spatial + seasonal signal.
KNN_CONTEXT_FEATURES = [
    "wind_speed", "rh", "temperature_2m",
    "viirs_ntl", "aer_ai",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
]

KNN_NEIGHBORS = 5   # tune if needed


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    log.info("Loaded  shape=%s  from %s", df.shape, path)
    return df


def drop_dead_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in DEAD_COLUMNS if c in df.columns]
    if present:
        df = df.drop(columns=present)
        log.info("Dropped 100%%-missing columns: %s", present)
    else:
        log.warning("Dead columns not found — skipping drop step.")
    return df


def impute_aod_by_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute AOD columns using KNN, applied PER STATION so that each
    station's own seasonal patterns guide the imputation — not values
    from distant parts of the city.
    """
    missing_before = df[AOD_COLUMNS].isnull().sum()
    log.info("Missing before imputation:\n%s", missing_before.to_string())

    stations = df["station"].unique()
    imputer  = KNNImputer(n_neighbors=KNN_NEIGHBORS)

    imputed_frames = []
    for station in stations:
        mask      = df["station"] == station
        sub       = df.loc[mask].copy()

        # Columns we actually feed to KNN (context + targets)
        cols_for_knn = KNN_CONTEXT_FEATURES + AOD_COLUMNS
        # Only keep context columns that exist and have no NaNs for this station
        safe_context = [
            c for c in KNN_CONTEXT_FEATURES
            if c in sub.columns and sub[c].isnull().sum() == 0
        ]
        cols_for_knn = safe_context + AOD_COLUMNS

        X = sub[cols_for_knn].values
        X_imputed = imputer.fit_transform(X)

        result = sub.copy()
        # Write back only the AOD columns (last 2 columns in our matrix)
        for i, col in enumerate(AOD_COLUMNS):
            result[col] = X_imputed[:, len(safe_context) + i]

        imputed_frames.append(result)
        missing_aod = sub[AOD_COLUMNS].isnull().sum().sum()
        if missing_aod:
            log.info("  Station %-25s  imputed %d AOD values", station, missing_aod)

    df_out = pd.concat(imputed_frames).sort_values(["date", "station"]).reset_index(drop=True)

    missing_after = df_out[AOD_COLUMNS].isnull().sum()
    log.info("Missing after imputation:\n%s", missing_after.to_string())

    remaining = missing_after.sum()
    if remaining:
        log.warning(
            "%d values still missing after KNN — filling with station-level "
            "monthly median as fallback.", remaining
        )
        for col in AOD_COLUMNS:
            median_fill = (
                df_out.groupby(["station", "month"])[col]
                .transform("median")
            )
            df_out[col] = df_out[col].fillna(median_fill)
        log.info("Fallback fill complete. Remaining NaNs: %d",
                 df_out[AOD_COLUMNS].isnull().sum().sum())

    return df_out


def validate(df: pd.DataFrame) -> None:
    """Sanity checks before saving."""
    assert df.isnull().sum().sum() == 0, (
        f"Dataset still has NaNs after imputation!\n{df.isnull().sum()[df.isnull().sum()>0]}"
    )
    for col in DEAD_COLUMNS:
        assert col not in df.columns, f"Dead column '{col}' still present!"
    log.info("✅ Validation passed — no NaNs, dead columns removed.")


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved  shape=%s  → %s", df.shape, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",  required=True,
        help="Path to the Phase 2 merged CSV (e.g. data/processed/merged_karachi_dataset.csv)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Where to save the cleaned CSV (e.g. data/processed/merged_clean.csv)"
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = load_data(input_path)
    df = drop_dead_columns(df)
    df = impute_aod_by_station(df)
    validate(df)
    save(df, output_path)

    log.info("─" * 60)
    log.info("Phase 3 Step 1 complete.")
    log.info("  Rows : %d", len(df))
    log.info("  Cols : %d  → %s", len(df.columns), list(df.columns))
    log.info("─" * 60)


if __name__ == "__main__":
    main()
