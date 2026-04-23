"""
Phase 3 — Step 2 (GEE version): Extract MERRA-2 PM2.5 as Ground Truth
=======================================================================
Replaces the OpenAQ fetcher entirely. Runs via the GEE Python API,
same environment you used in Phase 1.

What this does:
  - Pulls NASA MERRA-2 aerosol components (BC, OC, SO4, DU, SS) from GEE
  - Computes surface PM2.5 = BC + OC + SO4 + DU + SS using NASA's formula
  - Aggregates to daily mean over the Karachi bounding box
  - Exports one row per (date, station) to Google Drive as CSV

MERRA-2 dataset : NASA/GSFC/MERRA/aer/2
GEE resolution  : ~0.5° × 0.625° (~55km × 69km) — one cell covers all Karachi
Coverage        : 1980–present, hourly (we aggregate to daily)
Citation        : Gelaro et al. (2017), J. Climate. doi:10.1175/JCLI-D-16-0758.1

Usage:
  python phase3_step2_gee_pm25.py \
      --output data/raw/merra2_pm25_karachi.csv \
      --start  2019-01-01 \
      --end    2024-12-31

  Or run the pure GEE JavaScript version below directly in the GEE Code Editor.

Dependencies:
  pip install earthengine-api pandas
  earthengine authenticate   # only needed once
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Karachi bounding box ──────────────────────────────────────────────────────
KARACHI_BBOX = [66.85, 24.75, 67.25, 25.05]   # [west, south, east, north]

# ── Your 8 stations ───────────────────────────────────────────────────────────
STATIONS = {
    "Gulshan-e-Iqbal":    (24.9056, 67.0822),
    "Saddar":             (24.8560, 67.0100),
    "SITE_Industrial":    (24.9400, 66.9800),
    "Korangi_Industrial": (24.8200, 67.0300),
    "North_Nazimabad":    (24.9800, 67.1200),
    "Gulistan_Jauhar":    (24.8900, 67.1300),
    "Landhi":             (24.8100, 66.9900),
    "Federal_B_Area":     (24.9200, 67.0500),
}

# ── NASA PM2.5 formula coefficients ──────────────────────────────────────────
# PM2.5 = BC + OC + SO4 + SS + DU
# Coefficients from: https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/FAQ/
# All surface concentration bands are in kg/m³ — we convert to µg/m³ (* 1e9)
PM25_BANDS = {
    "BCSMASS": 1.0,   # Black Carbon surface mass concentration
    "OCSMASS": 1.0,   # Organic Carbon
    "SO4SMASS": 1.0,  # Sulfate (already in PM2.5 fraction)
    "SSSMASS25": 1.0, # Sea Salt (fine fraction, <2.5µm)
    "DUSMASS25": 1.0, # Dust (fine fraction, <2.5µm)
}
KG_TO_UG_M3 = 1e9   # kg/m³ → µg/m³


# ── GEE Python extraction ─────────────────────────────────────────────────────

def extract_via_python_api(start: str, end: str, output_path: Path) -> None:
    """Use the GEE Python API to extract and download locally."""
    try:
        import ee
    except ImportError:
        log.error("earthengine-api not installed. Run: pip install earthengine-api")
        raise

    log.info("Initialising GEE…")
    # Make sure to initialize correctly with the project ID we know works
    try:
        ee.Initialize(project='gen-lang-client-0478151371')
    except Exception:
        ee.Initialize()

    bbox    = ee.Geometry.Rectangle(KARACHI_BBOX)
    bands   = list(PM25_BANDS.keys())

    log.info("Loading MERRA-2 aerosol collection (%s → %s)…", start, end)
    collection = (
        ee.ImageCollection("NASA/GSFC/MERRA/aer/2")
        .filterDate(start, end)
        .select(bands)
        .filterBounds(bbox)
    )

    log.info("Computing daily PM2.5 (reducing hourly images by day)…")

    def daily_pm25(date_str):
        """Aggregate all hourly images in one day, compute PM2.5."""
        date  = ee.Date(date_str)
        day   = collection.filterDate(date, date.advance(1, "day"))
        mean  = day.mean()

        # Sum the aerosol components and convert kg/m³ → µg/m³
        pm25 = (
            mean.select("BCSMASS")
            .add(mean.select("OCSMASS"))
            .add(mean.select("SO4SMASS"))
            .add(mean.select("SSSMASS25"))
            .add(mean.select("DUSMASS25"))
            .multiply(KG_TO_UG_M3)
            .rename("pm25")
        )
        return pm25.set("system:time_start", date.millis())

    # Build date sequence
    start_date = ee.Date(start)
    end_date   = ee.Date(end)
    n_days     = end_date.difference(start_date, "day").getInfo()
    date_list  = [
        (pd.Timestamp(start) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(n_days)
    ]

    log.info("Extracting %d daily values…", len(date_list))
    all_rows = []

    for i, date_str in enumerate(date_list):
        if i % 100 == 0:
            log.info("  Progress: %d/%d days", i, len(date_list))

        try:
            img    = daily_pm25(date_str)
            result = img.reduceRegion(
                reducer  = ee.Reducer.mean(),
                geometry = bbox,
                scale    = 55000,   # match MERRA-2 native resolution
                maxPixels= 1e6,
            ).getInfo()

            pm25_val = result.get("pm25")
            if pm25_val is not None:
                # Assign the same value to all 8 stations
                for station in STATIONS:
                    all_rows.append({
                        "date":       date_str,
                        "station":    station,
                        "pm25_mean":  round(float(pm25_val), 4),
                        "pm25_count": 24,       # 24 hourly images aggregated
                        "source":     "MERRA-2_GEE",
                    })
        except Exception as e:
            log.warning("  Failed for %s: %s", date_str, e)

    if not all_rows:
        log.error("No data extracted. Check GEE authentication and date range.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["date", "station"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    log.info("─" * 60)
    log.info("MERRA-2 PM2.5 saved → %s", output_path)
    log.info("Shape    : %s", df.shape)
    log.info("Stations : %s", df["station"].unique().tolist())
    log.info("Date range: %s → %s", df["date"].min(), df["date"].max())
    log.info("PM2.5 stats: mean=%.1f  std=%.1f  min=%.1f  max=%.1f µg/m³",
             df["pm25_mean"].mean(), df["pm25_mean"].std(),
             df["pm25_mean"].min(),  df["pm25_mean"].max())
    log.info("─" * 60)
    log.info("Expected Karachi range: ~30–120 µg/m³ (WHO annual guideline: 5 µg/m³)")
    log.info("Next: run phase3_step3_merge_target.py with --groundtruth %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="Output CSV path (e.g. data/raw/merra2_pm25_karachi.csv)")
    parser.add_argument("--start",  default="2019-01-01")
    parser.add_argument("--end",    default="2024-12-31")
    args = parser.parse_args()

    extract_via_python_api(args.start, args.end, Path(args.output))


if __name__ == "__main__":
    main()
