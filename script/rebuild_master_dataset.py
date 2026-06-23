"""Rebuild `master_dataset.csv` with REAL PM2.5 ground truth.

Strict fallback chain (in order, no fabrication):
  1. OpenAQ v3 per-station PM2.5     → pm25_source = 'openaq_exact'
  2. US Consulate Karachi anchor     → pm25_source = 'openaq_us_consulate'
  3. MERRA-2 citywide scalar (GEE)   → pm25_source = 'merra2_citywide'
  4. otherwise                       → pm25_source = 'missing'

If < 30% of rows have any *real* (non-MERRA-2) ground truth, this script PAUSES
for 10 seconds with a loud warning before falling back to MERRA-2. The user
can Ctrl-C to abort.

Credentials are read from a .env file at the repo root (NOT committed). See
.env.example for the schema.

Usage:
  python script/rebuild_master_dataset.py [--dry-run] [--start 2019-01-01] [--end 2024-12-31]
"""
# ── .env loading (must be first) ─────────────────────────────────────────────
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
        # Do NOT log the contents — keys must stay out of stdout
    else:
        sys.stderr.write(
            f"[warn] No .env found at {_ENV_PATH}. Falling back to OS env vars.\n"
        )
except ImportError:
    # python-dotenv is optional; OS env vars still work
    pass

import argparse
import logging
import time
import subprocess

import pandas as pd

from stations_loader import STATION_NAMES, STATIONS_LIST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPT_DIR  = REPO_ROOT / "script"
# Real GEE exports live in notebooks/data/raw/ (they were created by an
# earlier run of run_data_collection.py that exported to Google Drive; the
# user manually placed them here). The orchestrator reads from this dir.
# If you re-run run_data_collection.py and want the orchestrator to use the
# freshly-queued exports, set RAW_DIR=... in the env or pass --raw-dir.
RAW_DIR     = REPO_ROOT / "notebooks" / "data" / "raw"
PROC_DIR    = REPO_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

OPENAQ_CSV    = RAW_DIR  / "openaq_pm25_karachi.csv"
MERRA2_CSV    = RAW_DIR  / "merra2_pm25_karachi.csv"
FEATURES_CSV  = PROC_DIR / "merged_karachi_dataset.csv"
CLEAN_CSV     = PROC_DIR / "merged_clean.csv"
MASTER_CSV    = PROC_DIR / "master_dataset.csv"
MODELING_CSV  = PROC_DIR / "modeling_dataset.csv"
COVERAGE_CSV  = PROC_DIR / "pm25_coverage_report.csv"

REAL_SOURCES = {"openaq_exact", "openaq_us_consulate"}
COVERAGE_THRESHOLD = 0.30   # below this → warn loudly and fall back to MERRA-2


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_step(cmd: list[str], desc: str) -> int:
    """Run a subprocess step. Log start/end. Return its exit code."""
    log.info("=" * 70)
    log.info("STEP: %s", desc)
    log.info("CMD : %s", " ".join(cmd))
    log.info("=" * 70)
    result = subprocess.run(cmd, check=False)
    log.info("→ %s exited with code %d", desc, result.returncode)
    return result.returncode


def _features_exist() -> bool:
    """Are there already GEE-exported raw CSVs we can skip GEE re-fetch for?"""
    if not RAW_DIR.exists():
        return False
    return any(RAW_DIR.glob("*.csv"))


def _step_1_features(args, dry_run: bool) -> bool:
    """Phase 1+2: collect features from GEE, then clean/impute.

    Skipped entirely if --skip-features and the file already exists."""
    if args.skip_features and CLEAN_CSV.exists():
        log.info("Skipping feature collection (--skip-features); reusing %s",
                 CLEAN_CSV)
        return True

    if not _features_exist():
        log.info("No GEE exports in %s — running run_data_collection.py", RAW_DIR)
        rc = _run_step(
            [sys.executable, str(SCRIPT_DIR / "run_data_collection.py"),
             "--start", args.start, "--end", args.end],
            "GEE data collection",
        )
        if rc != 0:
            log.error("GEE data collection failed (rc=%d). Aborting.", rc)
            return False
    else:
        log.info("Reusing existing GEE exports in %s", RAW_DIR)

    # merge GEE exports → merged_karachi_dataset.csv
    merged_out = FEATURES_CSV
    rc = _run_step(
        [sys.executable, str(SCRIPT_DIR / "merge_data.py"),
         "--raw-dir", str(RAW_DIR), "--out-path", str(merged_out)],
        "merge_data.py",
    )
    if rc != 0:
        log.error("merge_data.py failed (rc=%d). Aborting.", rc)
        return False

    if not merged_out.exists():
        log.error("Expected %s to exist after merge_data.py — aborting.", merged_out)
        return False

    if dry_run:
        log.info("[dry-run] Still running clean/impute (intermediate; not the canonical master).")

    # Clean & impute → merged_clean.csv
    rc = _run_step(
        [sys.executable, str(SCRIPT_DIR / "phase3_step1_clean_impute.py"),
         "--input", str(merged_out), "--output", str(CLEAN_CSV)],
        "phase3_step1_clean_impute.py",
    )
    return rc == 0 and CLEAN_CSV.exists()


def _step_2_openaq(args, dry_run: bool) -> pd.DataFrame:
    """Phase 2a: fetch OpenAQ per-station + US Consulate anchor.

    Reuses an existing OPENAQ_CSV if it has data and --refresh-openaq is not set
    (avoids paying 5+ minutes of API calls when the cache is fresh)."""
    refresh = getattr(args, "refresh_openaq", False)
    if OPENAQ_CSV.exists() and OPENAQ_CSV.stat().st_size > 0 and not refresh:
        try:
            existing = pd.read_csv(OPENAQ_CSV)
            if len(existing) > 0:
                log.info("Reusing existing OpenAQ cache: %s (%d rows)",
                         OPENAQ_CSV, len(existing))
                return existing
        except Exception as e:
            log.warning("Cache %s unreadable: %s — re-fetching.",
                        OPENAQ_CSV, e)

    cmd = [sys.executable, str(SCRIPT_DIR / "phase3_step2_fetch_groundtruth.py"),
           "--output", str(OPENAQ_CSV),
           "--start", args.start, "--end", args.end,
           "--min-readings", str(args.min_readings)]
    if os.environ.get("OPENAQ_KEY"):
        cmd += ["--api-key", os.environ["OPENAQ_KEY"]]
    rc = _run_step(cmd, "phase3_step2_fetch_groundtruth.py")
    if rc != 0 and not OPENAQ_CSV.exists():
        log.warning("OpenAQ fetch failed; will rely on MERRA-2 fallback.")

    if not OPENAQ_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(OPENAQ_CSV)
    log.info("OpenAQ rows: %d | stations: %s",
             len(df), sorted(df["station"].unique().tolist()))
    return df


def _step_2_merra2(args, dry_run: bool) -> pd.DataFrame:
    """Phase 2b: fetch MERRA-2 citywide (only when real coverage is too low).

    Reuses an existing MERRA2_CSV if it has data and --refresh-merra2 is not set."""
    refresh = getattr(args, "refresh_merra2", False)
    if MERRA2_CSV.exists() and MERRA2_CSV.stat().st_size > 0 and not refresh:
        try:
            existing = pd.read_csv(MERRA2_CSV)
            if len(existing) > 0:
                log.info("Reusing existing MERRA-2 cache: %s (%d rows)",
                         MERRA2_CSV, len(existing))
                return existing
        except Exception as e:
            log.warning("Cache %s unreadable: %s — re-fetching.",
                        MERRA2_CSV, e)

    rc = _run_step(
        [sys.executable, str(SCRIPT_DIR / "phase3_step2_gee_pm25.py"),
         "--output", str(MERRA2_CSV),
         "--start", args.start, "--end", args.end],
        "phase3_step2_gee_pm25.py",
    )
    if rc != 0:
        log.warning("MERRA-2 fetch failed (rc=%d).", rc)

    if not MERRA2_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(MERRA2_CSV)
    log.info("MERRA-2 rows: %d | scope: %s",
             len(df), df["scope"].iloc[0] if "scope" in df.columns else "?")
    return df


def _build_combined_gt(openaq_df: pd.DataFrame, merra2_df: pd.DataFrame) -> pd.DataFrame:
    """Stack OpenAQ and MERRA-2 into a single ground-truth CSV the merger can
    read. The merger dispatches on the `pm25_source` column:
      • 'openaq_exact'         → per-station (date, station) join
      • 'openaq_us_consulate'  → date-only broadcast
      • 'merra2_citywide'      → date-only broadcast
    """
    parts = []
    if not openaq_df.empty:
        oa = openaq_df.copy()
        if "pm25_mean" not in oa.columns and "pm25" in oa.columns:
            oa = oa.rename(columns={"pm25": "pm25_mean"})
        # The OpenAQ script already writes pm25_source per row; trust it.
        parts.append(oa)

    if not merra2_df.empty:
        m2 = merra2_df.copy()
        if "pm25_mean" not in m2.columns and "pm25" in m2.columns:
            m2 = m2.rename(columns={"pm25": "pm25_mean"})
        # The MERRA-2 script writes `source='MERRA-2_GEE'` and `scope='citywide'`
        # but does NOT populate `pm25_source`. Stamp it here so the merger
        # can dispatch on the standard column name.
        if "pm25_source" not in m2.columns or m2["pm25_source"].isna().all():
            m2["pm25_source"] = "merra2_citywide"
        else:
            m2["pm25_source"] = m2["pm25_source"].fillna("merra2_citywide")
        parts.append(m2)

    if not parts:
        return pd.DataFrame(columns=["date", "pm25_mean", "pm25_source"])

    combined = pd.concat(parts, ignore_index=True)
    return combined


def _step_3_merge(args, dry_run: bool) -> Path | None:
    """Phase 3: merge features × ground truth → master_dataset.csv."""
    if not CLEAN_CSV.exists():
        log.error("Cleaned features not found at %s — aborting.", CLEAN_CSV)
        return None

    openaq_df = _step_2_openaq(args, dry_run)
    merra2_df = pd.DataFrame()
    features   = pd.read_csv(CLEAN_CSV, parse_dates=["date"])
    days       = (features["date"].max() - features["date"].min()).days + 1
    expected   = len(STATION_NAMES) * days

    real_rows = 0 if openaq_df.empty else int(
        openaq_df["pm25_source"].isin(REAL_SOURCES).sum()
    )
    real_pct = (real_rows / expected * 100) if expected else 0.0
    log.info("─" * 70)
    log.info("Real OpenAQ ground-truth coverage: %d / %d rows (%.1f%%)",
             real_rows, expected, real_pct)
    log.info("─" * 70)

    if real_pct < (COVERAGE_THRESHOLD * 100):
        log.warning(
            "Real coverage (%.1f%%) is below the %.0f%% threshold.",
            real_pct, COVERAGE_THRESHOLD * 100
        )
        _loud_low_coverage_warning(real_pct)
        log.info("Falling back to MERRA-2 citywide scalar for the missing rows.")
        merra2_df = _step_2_merra2(args, dry_run)
    else:
        log.info("Real coverage OK — skipping MERRA-2 fallback.")

    combined_gt = _build_combined_gt(openaq_df, merra2_df)
    combined_path = RAW_DIR / "pm25_combined_ground_truth.csv"
    combined_gt.to_csv(combined_path, index=False)
    log.info("Combined ground truth written → %s (%d rows)",
             combined_path, len(combined_gt))

    if dry_run:
        log.info("[dry-run] Would run phase3_step3_merge_target.py now.")
        log.info("[dry-run] Skipping final write to %s", MASTER_CSV)
        return None

    rc = _run_step(
        [sys.executable, str(SCRIPT_DIR / "phase3_step3_merge_target.py"),
         "--features",    str(CLEAN_CSV),
         "--groundtruth", str(combined_path),
         "--output",      str(MASTER_CSV)],
        "phase3_step3_merge_target.py",
    )
    if rc != 0 or not MASTER_CSV.exists():
        log.error("Merge step failed.")
        return None
    return MASTER_CSV


def _loud_low_coverage_warning(real_pct: float) -> None:
    """10-second abort window before MERRA-2 fallback."""
    sys.stderr.write("\n")
    sys.stderr.write("*" * 70 + "\n")
    sys.stderr.write("***  WARNING: LOW REAL GROUND-TRUTH COVERAGE  ***\n")
    sys.stderr.write("*" * 70 + "\n")
    sys.stderr.write(
        f"Only {real_pct:.1f}% of rows have REAL ground truth\n"
        f"(OpenAQ per-station or US Consulate anchor).\n\n"
        f"The remaining {100 - real_pct:.1f}% of rows will be FILLED with\n"
        f"MERRA-2 citywide scalar (same value for all 8 stations).\n\n"
        f"CONSEQUENCES:\n"
        f"  • Per-station metrics, spatial analysis, and LISA hotspots will be\n"
        f"    MEANINGLESS for MERRA-2-filled rows.\n"
        f"  • The dataset is real (not fabricated), but it is spatially coarse.\n\n"
        f"REMEDIATION BEFORE CONTINUING:\n"
        f"  (a) Register for a free OpenAQ key at\n"
        f"      https://explore.openaq.org/register  (~10× faster fetch).\n"
        f"  (b) Run `python script/phase3_step2_fetch_groundtruth.py` alone to\n"
        f"      inspect per-station coverage before continuing.\n\n"
        f"  Press Ctrl-C within 10 seconds to abort, or wait to continue\n"
        f"  with the MERRA-2 fallback.\n"
    )
    sys.stderr.write("*" * 70 + "\n")
    sys.stderr.flush()
    for i in range(10, 0, -1):
        sys.stderr.write(f"\r  Continuing in {i:2d}s… (Ctrl-C to abort) ")
        sys.stderr.flush()
        time.sleep(1)
    sys.stderr.write("\r  Continuing.                                    \n")
    sys.stderr.flush()


def _step_4_verify(master_path: Path) -> bool:
    """Phase 4: emit modeling_dataset.csv (no NaN target rows) + report."""
    df = pd.read_csv(master_path)
    if "pm25" not in df.columns:
        log.error("master_dataset.csv has no 'pm25' column!")
        return False

    real_pct = 100 * df["pm25_source"].isin(REAL_SOURCES | {"merra2_citywide"}).mean()
    log.info("─" * 70)
    log.info("Final master_dataset.csv:")
    log.info("  Rows              : %d", len(df))
    log.info("  Columns           : %d", len(df.columns))
    log.info("  pm25_source dist  :")
    for src, n in df["pm25_source"].value_counts().items():
        log.info("    %-25s %5d (%.1f%%)", src, n, 100 * n / len(df))
    log.info("  PM2.5 (filled)    : mean=%.1f  std=%.1f  min=%.1f  max=%.1f µg/m³",
             df["pm25"].mean(), df["pm25"].std(),
             df["pm25"].min(), df["pm25"].max())
    log.info("─" * 70)

    # Modeling dataset = drop rows where pm25 is missing
    model_df = df[df["pm25"].notna() & (df["pm25_source"] != "missing")].copy()
    model_df.to_csv(MODELING_CSV, index=False)
    log.info("modeling_dataset.csv → %s (%d rows with valid target)",
             MODELING_CSV, len(model_df))

    # Coverage report
    coverage = (
        df.groupby("station")["pm25_source"]
        .value_counts()
        .unstack(fill_value=0)
    )
    coverage.to_csv(COVERAGE_CSV)
    log.info("pm25_coverage_report.csv → %s", COVERAGE_CSV)

    if (df["pm25_source"] == "missing").all():
        log.error(
            "ALL rows have pm25_source='missing'. Pipeline produced NO real "
            "ground truth. Check (1) .env OPENAQ_KEY, (2) GEE auth, (3) network."
        )
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--min-readings", type=int, default=3,
                        help="Min hourly OpenAQ readings per day (default 3).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run fetches but DO NOT overwrite master_dataset.csv.")
    parser.add_argument("--skip-features", action="store_true",
                        help="Reuse existing merged_clean.csv; do not re-fetch GEE.")
    parser.add_argument("--refresh-openaq", action="store_true",
                        help="Re-fetch OpenAQ data even if a cached CSV exists.")
    parser.add_argument("--refresh-merra2", action="store_true",
                        help="Re-fetch MERRA-2 data even if a cached CSV exists.")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  Rebuild master_dataset.csv with REAL PM2.5 ground truth")
    log.info("  Range: %s → %s", args.start, args.end)
    log.info("  Stations: %d", len(STATION_NAMES))
    log.info("  OPENAQ_KEY set: %s",
             "yes (length=%d)" % len(os.environ["OPENAQ_KEY"])
             if os.environ.get("OPENAQ_KEY") else "no (anonymous tier)")
    log.info("  GEE_PROJECT   : %s", os.environ.get("GEE_PROJECT", "<unset>"))
    log.info("=" * 70)

    if not _step_1_features(args, args.dry_run):
        sys.exit(1)

    master_path = _step_3_merge(args, args.dry_run)
    if master_path is None:
        if args.dry_run:
            log.info("Dry run complete (no master_dataset.csv written).")
            sys.exit(0)
        sys.exit(1)

    if not _step_4_verify(master_path):
        sys.exit(1)

    log.info("")
    log.info("✅  DONE. Real master_dataset.csv and modeling_dataset.csv are ready.")
    log.info("    master     → %s", MASTER_CSV)
    log.info("    modeling   → %s", MODELING_CSV)
    log.info("    coverage   → %s", COVERAGE_CSV)


if __name__ == "__main__":
    main()
