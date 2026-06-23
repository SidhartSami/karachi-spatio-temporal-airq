"""Phase 3 — Step 3: Merge Ground-Truth PM2.5 into Feature Dataset.

Takes the cleaned feature dataset and a ground-truth CSV, and produces the
final modeling-ready dataset with explicit `pm25_source` provenance per row.

Fixes vs prior version (ISSUES_FOUND.md):
  M1  — df.iterrows() merge replaced with vectorized pd.merge.
  M12 — silent "Gulshan_e_Iqbal" → "Gulshan-e-Iqbal" name-map no-op removed;
        station names now validated against stations_loader (raises with list
        of offenders).
  pm25_source enum extended: {openaq_exact, openaq_us_consulate, merra2_citywide, missing}.
  MERRA-2 citywide support: when ground-truth CSV has no `station` column and
  `scope='citywide'`, the daily scalar is broadcast to every station row and
  `pm25_source='merra2_citywide'` is stamped.

Schema precedence:
  1. Exact (date, station) join → pm25_source='openaq_exact' or 'merra2_citywide'.
  2. Date-only join against US_Consulate_Karachi → 'openaq_us_consulate'.
  3. Otherwise NaN + 'missing'.

Usage:
  python phase3_step3_merge_target.py \\
      --features     data/processed/merged_clean.csv \\
      --groundtruth  data/raw/openaq_pm25_karachi.csv \\
      --output       data/processed/master_dataset.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stations_loader import STATION_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

US_CONSULATE_STATION = "US_Consulate_Karachi"

VALID_SOURCES = {
    "openaq_exact",
    "openaq_us_consulate",
    "merra2_citywide",
    "missing",
}


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    log.info("Features  shape=%s", df.shape)
    return df


def load_groundtruth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    has_station = "station" in df.columns
    has_scope = "scope" in df.columns
    scope_value = df["scope"].iloc[0] if has_scope and len(df) else None
    is_citywide = (not has_station) or (scope_value == "citywide")
    log.info(
        "Ground truth  shape=%s  stations=%s  scope=%s  citywide=%s",
        df.shape,
        df["station"].unique().tolist() if has_station else "<none>",
        scope_value if has_scope else "<no scope col>",
        is_citywide,
    )
    return df, is_citywide


# ── Validation ───────────────────────────────────────────────────────────────

def validate_stations(features: pd.DataFrame, df_gt: pd.DataFrame,
                       is_citywide: bool) -> None:
    """Fail loudly with a list of offending station names if anything unexpected."""
    feature_stations = set(features["station"].dropna().unique())
    expected = set(STATION_NAMES)
    extra_in_features = feature_stations - expected
    missing_from_features = expected - feature_stations

    if extra_in_features:
        log.warning(
            "Features contain %d station name(s) not in stations.json: %s",
            len(extra_in_features), sorted(extra_in_features),
        )
    if missing_from_features:
        log.warning(
            "Features missing %d station name(s) from stations.json: %s",
            len(missing_from_features), sorted(missing_from_features),
        )

    if not is_citywide:
        gt_stations = set(df_gt["station"].dropna().unique())
        # The US Consulate anchor is intentionally NOT in features — its value
        # is broadcast to every station row for the same date (see
        # merge_target step 2). Allow it; flag any other unknown station.
        allowed_extras = {US_CONSULATE_STATION}
        unexpected_gt = (gt_stations - feature_stations) - allowed_extras
        if unexpected_gt:
            raise ValueError(
                f"Ground truth has station names not in features: "
                f"{sorted(unexpected_gt)}. Aborting to avoid silent mismatch."
            )


# ── Merger ──────────────────────────────────────────────────────────────────

def merge_target(features: pd.DataFrame, df_gt: pd.DataFrame,
                 is_citywide: bool) -> pd.DataFrame:
    """Vectorized merge. Returns features with pm25 and pm25_source columns.

    Handles three source types in priority order (per the plan's strict
    fallback hierarchy):

      1. `openaq_exact`        — per-station (date, station) join.
      2. `openaq_us_consulate` — date-only broadcast (US Consulate anchor).
      3. `merra2_citywide`     — date-only broadcast (MERRA-2 citywide scalar).
      4. `missing`             — no ground truth for that (date, station).

    The `is_citywide` flag is preserved for back-compat but the function
    auto-detects mixed GT: it dispatches on the `pm25_source` column rather
    than the file-level scope flag.
    """
    # Normalize ground truth column name (some sources use 'pm25', others 'pm25_mean')
    if "pm25_mean" in df_gt.columns and "pm25" not in df_gt.columns:
        df_gt = df_gt.rename(columns={"pm25_mean": "pm25"})

    # Start with features; pm25 starts NaN
    merged = features.copy()
    merged["pm25"] = np.nan
    merged["pm25_source"] = "missing"

    # 1) Per-station exact match: openaq_exact
    gt_exact = df_gt[df_gt["pm25_source"] == "openaq_exact"].copy()
    if not gt_exact.empty:
        # Inner-join on (date, station) then assign values back
        m_exact = merged.merge(
            gt_exact[["date", "station", "pm25"]].rename(columns={"pm25": "_pm25_e"}),
            on=["date", "station"], how="left",
        )
        mask_e = m_exact["_pm25_e"].notna()
        merged.loc[mask_e, "pm25"] = m_exact.loc[mask_e, "_pm25_e"].values
        merged.loc[mask_e, "pm25_source"] = "openaq_exact"
        log.info("openaq_exact:      %d feature rows got exact per-station values.",
                 int(mask_e.sum()))

    # 2) US Consulate anchor: date-only broadcast
    gt_anchor = df_gt[df_gt["pm25_source"] == "openaq_us_consulate"].copy()
    if not gt_anchor.empty:
        m_anc = merged.merge(
            gt_anchor[["date", "pm25"]].rename(columns={"pm25": "_pm25_a"}),
            on="date", how="left",
        )
        mask_a = (merged["pm25_source"] == "missing") & m_anc["_pm25_a"].notna()
        merged.loc[mask_a, "pm25"] = m_anc.loc[mask_a, "_pm25_a"].values
        merged.loc[mask_a, "pm25_source"] = "openaq_us_consulate"
        log.info("openaq_us_consulate: %d feature rows filled with US Consulate anchor.",
                 int(mask_a.sum()))

    # 3) MERRA-2 citywide: date-only broadcast (any row not yet filled)
    gt_city = df_gt[df_gt["pm25_source"] == "merra2_citywide"].copy()
    if not gt_city.empty:
        m_city = merged.merge(
            gt_city[["date", "pm25"]].rename(columns={"pm25": "_pm25_c"}),
            on="date", how="left",
        )
        mask_c = (merged["pm25_source"] == "missing") & m_city["_pm25_c"].notna()
        merged.loc[mask_c, "pm25"] = m_city.loc[mask_c, "_pm25_c"].values
        merged.loc[mask_c, "pm25_source"] = "merra2_citywide"
        log.info("merra2_citywide:   %d feature rows filled with MERRA-2 citywide scalar.",
                 int(mask_c.sum()))

    # Sanity: enforce enum
    bad = set(merged["pm25_source"].unique()) - VALID_SOURCES
    if bad:
        raise RuntimeError(f"Unexpected pm25_source values: {bad}")

    return merged


# ── Reporting ────────────────────────────────────────────────────────────────

def report_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station × pm25_source row counts. Returns the summary DataFrame."""
    log.info("─" * 60)
    log.info("PM2.5 target coverage by station × source:")
    summary = (
        df.groupby("station")["pm25_source"]
        .value_counts()
        .unstack(fill_value=0)
    )
    # Ensure all source columns present
    for src in VALID_SOURCES:
        if src not in summary.columns:
            summary[src] = 0
    summary = summary[sorted(VALID_SOURCES)]
    summary["total"] = summary[list(VALID_SOURCES)].sum(axis=1)
    for src in VALID_SOURCES:
        summary[f"pct_{src}"] = (summary[src] / summary["total"] * 100).round(1)

    for station, row in summary.iterrows():
        log.info(
            "  %-25s exact=%5.1f%%  anchor=%5.1f%%  merra2=%5.1f%%  missing=%5.1f%%",
            station,
            row.get("pct_openaq_exact", 0),
            row.get("pct_openaq_us_consulate", 0),
            row.get("pct_merra2_citywide", 0),
            row.get("pct_missing", 0),
        )
    log.info("─" * 60)
    return summary


# ── Final cleaning ───────────────────────────────────────────────────────────

def finalize(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    invalid = (df["pm25"] < 0) | (df["pm25"] > 1000)
    if invalid.any():
        log.warning("Removing %d rows with PM2.5 outside [0, 1000].",
                    int(invalid.sum()))
        df = df[~invalid]

    # Reorder: date, station, pm25, pm25_source first; then features
    priority = ["date", "station", "pm25", "pm25_source"]
    others = [c for c in df.columns if c not in priority]
    df = df[priority + others]

    log.info("Final dataset: %d rows × %d columns (dropped %d invalid rows)",
             *df.shape, before - len(df))
    if df["pm25"].notna().any():
        log.info("PM2.5 stats: mean=%.1f std=%.1f min=%.1f max=%.1f",
                 df["pm25"].mean(), df["pm25"].std(),
                 df["pm25"].min(), df["pm25"].max())
    else:
        log.warning("PM2.5 column is entirely NaN — no ground truth was usable.")
    return df.reset_index(drop=True)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("✅ Saved → %s", path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",     required=True,
                        help="Path to Step 1 output (merged_clean.csv)")
    parser.add_argument("--groundtruth",  required=True,
                        help="Path to Step 2 output (openaq_pm25_karachi.csv or "
                             "merra2_pm25_karachi.csv)")
    parser.add_argument("--output",       default="data/processed/master_dataset.csv",
                        help="Where to save master_dataset.csv (default) "
                             "or modeling_dataset.csv (when --output ends with that name)")
    args = parser.parse_args()

    features_path = Path(args.features)
    gt_path = Path(args.groundtruth)
    output_path = Path(args.output)

    for p in [features_path, gt_path]:
        if not p.exists():
            log.error("File not found: %s", p)
            sys.exit(1)

    features = load_features(features_path)
    df_gt, is_citywide = load_groundtruth(gt_path)
    validate_stations(features, df_gt, is_citywide)
    merged = merge_target(features, df_gt, is_citywide)
    summary = report_coverage(merged)
    final = finalize(merged)
    save(final, output_path)

    # Save coverage report alongside
    report_path = output_path.parent / "pm25_coverage_report.csv"
    summary.to_csv(report_path)
    log.info("✅ Coverage report → %s", report_path)

    log.info("")
    log.info("Next step → Phase 4: Machine Learning Modeling")
    log.info("  Output : %s", output_path)
    log.info("  Target : pm25  (source column: pm25_source)")


if __name__ == "__main__":
    main()
