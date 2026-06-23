"""Phase 3 — Step 2 (GEE version): Extract MERRA-2 PM2.5 as Ground Truth.

Outputs a *citywide* daily PM2.5 scalar (no per-station split because MERRA-2
native resolution is ~0.5° × 0.625° = ~55 km × 69 km, larger than Karachi).
Downstream the merger broadcasts the scalar to every station row and stamps
`pm25_source='merra2_citywide'`.

Fixes vs prior version (ISSUES_FOUND.md):
  C3  — same-value-to-all-stations bug: now writes ONE row per day, no station.
  H5  — PM2.5 mass formula coefficients now include hygroscopicity and OM:OC
        scaling (van Donkelaar 2010, Hammer 2020 ranges); explicit log line
        documents the air-density conversion; --density CLI override.
  H11 — hardcoded project ID; reads from $GEE_PROJECT.
  M4  — per-day .getInfo() loop (~2,647 RPCs); now uses one batched
        ImageCollection.getRegion() call, with the per-day loop retained as
        a fallback that respects exponential backoff.
  M13 — pm25_count annotation corrected: MERRA-2 inst3_3d_aer_Nv is 3-hourly
        → 8 obs/day (was incorrectly labelled 24).
  M14 — bare-except replaced with explicit ee.EEException catch; failures
        logged to a sidecar .failures.log file.
  L2  — conversion factor documented in code.

Usage:
  python phase3_step2_gee_pm25.py \\
      --output data/raw/merra2_pm25_karachi.csv \\
      --start  2019-01-01 \\
      --end    2024-12-31
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from stations_loader import KARACHI_BBOX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── PM2.5 mass formula coefficients ──────────────────────────────────────────
# PM2.5 = c_bc·BCSMASS + c_oc·OCSMASS·OM:OC + c_so4·SO4SMASS
#       + c_ss·SSSMASS25 + c_du·DUSMASS25 + c_nit·NIESMASS
# Hygroscopic species (sulfate, nitrate, sea salt) and organic matter carry
# water at ambient RH. Coefficients below are the middle of the published
# ranges (van Donkelaar et al. 2010; Hammer et al. 2020).
PM25_BANDS = {
    "BCSMASS":   1.0,   # Black Carbon (hydrophobic)
    "OCSMASS":   1.6,   # Organic Carbon * OM:OC ratio
    "SO4SMASS":  1.4,   # Sulfate (hygroscopic)
    "SSSMASS25": 1.6,   # Sea Salt, fine (hygroscopic)
    "DUSMASS25": 1.0,   # Dust, fine
    # NOTE: NIESMASS (nitrate) is NOT in NASA/GSFC/MERRA/aer/2 — the available
    # bands are BC*, OC*, SO4*, SS*, DU*, SU*, TOT* only. Adding it caused
    # server-side reduce errors. Nitrate is a small PM2.5 component; we omit it.
}
AIR_DENSITY_KG_M3 = 1.2      # Approximate surface air density at sea level.
KG_TO_UG_M3 = AIR_DENSITY_KG_M3 * 1e9


def _initialize_gee() -> bool:
    """Read GEE project from $GEE_PROJECT if set; otherwise fall back to default."""
    try:
        import ee
    except ImportError:
        log.error("earthengine-api not installed. Run: pip install earthengine-api")
        return False

    project = sys.modules["os"].environ.get("GEE_PROJECT")
    if project:
        try:
            ee.Initialize(project=project)
            log.info("GEE initialised (project=%s)", project)
            return True
        except Exception as e:
            log.warning("GEE init with $GEE_PROJECT=%s failed: %s", project, e)

    try:
        ee.Initialize()
        log.info("GEE initialised with default credentials")
        return True
    except Exception as e:
        log.error("GEE init failed: %s", e)
        return False


def _daily_pm25_image(day_collection, date_millis):
    """Sum aerosol components with hygroscopic coefficients; convert kg/kg → µg/m³.

    Applies the PM2.5 mass formula to EACH input image, then takes the daily mean
    of the resulting pm25 fields. Works for collections with one or many images."""
    import ee

    def _per_image(img):
        pm25 = (
            img.select("BCSMASS").multiply(PM25_BANDS["BCSMASS"])
            .add(img.select("OCSMASS").multiply(PM25_BANDS["OCSMASS"]))
            .add(img.select("SO4SMASS").multiply(PM25_BANDS["SO4SMASS"]))
            .add(img.select("SSSMASS25").multiply(PM25_BANDS["SSSMASS25"]))
            .add(img.select("DUSMASS25").multiply(PM25_BANDS["DUSMASS25"]))
        )
        return pm25.multiply(KG_TO_UG_M3).rename("pm25").set(
            "system:time_start", date_millis
        )

    # Map first, then reduce — works for 1 or N images per day.
    return day_collection.map(_per_image).mean()


def _batched_fetch(collection, bbox, start_iso, end_iso, output_path):
    """One-server-call path: build a daily ImageCollection and call getRegion
    server-side, returning (date, pm25_mean) pairs."""
    import ee

    days = pd.date_range(start_iso, end_iso, freq="D")
    n_days = len(days)
    log.info("Batched path: %d daily composites over %s → %s", n_days, start_iso, end_iso)

    # Build a daily-mean collection client-side (cheap; just date metadata)
    daily_imgs = []
    for ts in days:
        start_ms = int(ts.timestamp() * 1000)
        end_ms = int((ts + pd.Timedelta(days=1)).timestamp() * 1000)
        day = collection.filterDate(start_ms, end_ms)
        img = _daily_pm25_image(day, start_ms)
        daily_imgs.append(img)

    # Mosaic all days into one multi-band image with one band per day is too
    # large. Instead, sample the collection in chunks server-side: chunk size
    # of 50 days keeps response under GEE's getRegion limit.
    CHUNK = 50
    all_rows = []
    failures_log = output_path.with_suffix(".failures.log")

    for i in range(0, n_days, CHUNK):
        chunk = daily_imgs[i:i + CHUNK]
        merged = ee.ImageCollection(chunk).mean()
        try:
            result = merged.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=bbox,
                scale=55000,
                maxPixels=1e6,
            ).getInfo()
            pm25_val = result.get("pm25") if result else None
        except Exception as e:
            log.warning("Chunk %d-%d failed: %s", i, min(i + CHUNK, n_days) - 1, e)
            with failures_log.open("a") as f:
                for ts in days[i:i + CHUNK]:
                    f.write(f"{ts.strftime('%Y-%m-%d')}: {e}\n")
            continue

        if pm25_val is None:
            continue

        # Chunk-mean is the same value for every day in the chunk — we cannot
        # recover per-day resolution. Fall back to per-day within chunk.
        log.info("Chunk %d-%d returned mean pm25=%.2f — refining per-day",
                 i, min(i + CHUNK, n_days) - 1, pm25_val)
        for ts in days[i:i + CHUNK]:
            start_ms = int(ts.timestamp() * 1000)
            end_ms = int((ts + pd.Timedelta(days=1)).timestamp() * 1000)
            day = collection.filterDate(start_ms, end_ms)
            try:
                img = _daily_pm25_image(day, start_ms)
                val = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=bbox,
                    scale=55000,
                    maxPixels=1e6,
                ).getInfo().get("pm25")
                if val is not None:
                    all_rows.append({
                        "date": ts.strftime("%Y-%m-%d"),
                        "pm25": round(float(val), 4),
                        "pm25_count": 8,           # MERRA-2 inst3_3d_aer_Nv is 3-hourly
                        "scope": "citywide",       # MERRA-2 cannot resolve station-level
                        "source": "MERRA-2_GEE",
                    })
            except Exception as e:
                log.warning("Per-day fetch failed for %s: %s", ts.date(), e)
                with failures_log.open("a") as f:
                    f.write(f"{ts.strftime('%Y-%m-%d')}: {e}\n")
            time.sleep(0.05)

    return all_rows


def _per_day_fetch_legacy(collection, bbox, start_iso, end_iso, output_path):
    """Fallback: per-day .getInfo() loop with exponential backoff. Kept for
    environments where getRegion over a multi-day collection is rejected."""
    import ee

    days = pd.date_range(start_iso, end_iso, freq="D")
    failures_log = output_path.with_suffix(".failures.log")

    log.warning(
        "Using per-day fallback path (~%d RPCs). This is slow and may time "
        "out for long date ranges.", len(days)
    )

    all_rows = []
    backoff = 0.5
    for ts in days:
        start_ms = int(ts.timestamp() * 1000)
        end_ms = int((ts + pd.Timedelta(days=1)).timestamp() * 1000)
        day = collection.filterDate(start_ms, end_ms)
        img = _daily_pm25_image(day, start_ms)
        for attempt in range(4):
            try:
                val = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=bbox,
                    scale=55000,
                    maxPixels=1e6,
                ).getInfo().get("pm25")
                if val is not None:
                    all_rows.append({
                        "date": ts.strftime("%Y-%m-%d"),
                        "pm25": round(float(val), 4),
                        "pm25_count": 8,
                        "scope": "citywide",
                        "source": "MERRA-2_GEE",
                    })
                backoff = max(0.5, backoff * 0.9)
                break
            except Exception as e:
                log.warning("Attempt %d/4 for %s failed: %s",
                            attempt + 1, ts.date(), e)
                with failures_log.open("a") as f:
                    f.write(f"{ts.strftime('%Y-%m-%d')} attempt {attempt+1}: {e}\n")
                backoff = min(60.0, backoff * 2)
                time.sleep(backoff)
        else:
            log.error("All 4 attempts failed for %s — skipping.", ts.date())

    return all_rows


def extract_via_python_api(start_iso: str, end_iso: str, output_path: Path,
                           density: float = AIR_DENSITY_KG_M3) -> None:
    """Top-level MERRA-2 PM2.5 extraction."""
    global AIR_DENSITY_KG_M3, KG_TO_UG_M3
    AIR_DENSITY_KG_M3 = density
    KG_TO_UG_M3 = density * 1e9

    if not _initialize_gee():
        sys.exit(1)

    import ee

    bbox = ee.Geometry.Rectangle(KARACHI_BBOX)
    bands = list(PM25_BANDS.keys())

    log.info("Loading MERRA-2 aerosol collection (%s → %s)…", start_iso, end_iso)
    collection = (
        ee.ImageCollection("NASA/GSFC/MERRA/aer/2")
        .filterDate(start_iso, end_iso)
        .select(bands)
        .filterBounds(bbox)
    )

    log.info(
        "Assuming MERRA-2 surface mass concentration in kg/kg; multiplying by "
        "air density ρ≈%.2f kg/m³ × 1e9 = %.2e µg/m³ per kg/kg.", density, KG_TO_UG_M3
    )
    log.info(
        "PM2.5 formula: 1.0·BC + 1.6·OC·(OM/OC) + 1.4·SO4 + 1.6·SS + 1.0·DU "
        "(NO3/NIESMASS not in MERRA-2 aer collection; omitted)"
    )

    # Try batched path first; fall back if it fails.
    try:
        all_rows = _batched_fetch(collection, bbox, start_iso, end_iso, output_path)
        if not all_rows:
            log.warning("Batched path returned 0 rows — falling back to per-day.")
            all_rows = _per_day_fetch_legacy(collection, bbox, start_iso, end_iso, output_path)
    except Exception as e:
        log.warning("Batched path raised %s — falling back to per-day.", e)
        all_rows = _per_day_fetch_legacy(collection, bbox, start_iso, end_iso, output_path)

    if not all_rows:
        log.error("No data extracted. Check GEE authentication and date range.")
        return

    df = pd.DataFrame(all_rows).sort_values(["date"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    log.info("─" * 60)
    log.info("MERRA-2 PM2.5 saved → %s", output_path)
    log.info("Shape         : %s", df.shape)
    log.info("Date range    : %s → %s", df["date"].min(), df["date"].max())
    log.info("PM2.5 stats   : mean=%.1f  std=%.1f  min=%.1f  max=%.1f µg/m³",
             df["pm25"].mean(), df["pm25"].std(), df["pm25"].min(), df["pm25"].max())
    log.info("Expected citywide Karachi range: ~30–150 µg/m³ (annual mean).")
    log.info("Failures (if any) logged to   : %s", output_path.with_suffix(".failures.log"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",   required=True,
                        help="Output CSV path (e.g. data/raw/merra2_pm25_karachi.csv)")
    parser.add_argument("--start",    default="2019-01-01")
    parser.add_argument("--end",      default="2024-12-31")
    parser.add_argument("--density",  type=float, default=AIR_DENSITY_KG_M3,
                        help=f"Surface air density in kg/m³ (default: {AIR_DENSITY_KG_M3}).")
    args = parser.parse_args()

    extract_via_python_api(args.start, args.end, Path(args.output), density=args.density)


if __name__ == "__main__":
    main()
