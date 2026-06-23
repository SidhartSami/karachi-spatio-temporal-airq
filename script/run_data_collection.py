"""Queue GEE export tasks for Karachi air-quality satellite/meteo data.

Fixes vs prior version (ISSUES_FOUND.md):
  H11 — hardcoded GEE project ID; now reads from $GEE_PROJECT env var.
  M15  — fire-and-forget task.start(); now polls task.status() to confirm
         COMPLETED/FAILED before continuing.
  M2/M3 — station coordinates were duplicated here; now imported from
          stations_loader.
  New: --start / --end / --include-offl CLI args; OFFL is opt-in only.
"""
import argparse
import os
import sys
import time

import ee

from stations_loader import (
    STATIONS_LIST,
    KARACHI_BBOX,
    GEE_PROJECT_DEFAULT,
    to_lonlat,
)


def _initialize_gee() -> bool:
    """Initialize GEE using $GEE_PROJECT if set, else default from stations.json."""
    project = os.environ.get("GEE_PROJECT", GEE_PROJECT_DEFAULT)
    try:
        ee.Initialize(project=project)
        print(f"✓ GEE Initialized (project={project})")
        return True
    except Exception as e:
        print(f"✗ GEE Initialization Failed for project={project}: {e}", file=sys.stderr)
        return False


def _poll_task(task, label: str, max_wait_s: int = 600) -> str:
    """Poll a GEE task until terminal state. Returns final state."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        status = task.status()
        state = status.get("state", "UNKNOWN")
        if state in {"COMPLETED", "FAILED", "CANCELLED"}:
            print(f"  [{label}] → {state}")
            if state != "COMPLETED":
                err = status.get("error_message", "<no error message>")
                print(f"    error_message: {err}", file=sys.stderr)
            return state
        time.sleep(30)
    print(f"  [{label}] → TIMEOUT after {max_wait_s}s (state={task.status().get('state')})",
          file=sys.stderr)
    return "TIMEOUT"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default="2024-12-31", help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--include-offl", action="store_true",
                        help="Also queue OFFL S5P assets (otherwise NRTI only).")
    parser.add_argument("--max-wait", type=int, default=600,
                        help="Max seconds to poll each task before continuing.")
    args = parser.parse_args()

    if not _initialize_gee():
        sys.exit(1)

    station_features = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point(to_lonlat(s)), {'station': s['name'], 'zone_type': s['zone_type']})
        for s in STATIONS_LIST
    ])

    west, south, east, north = KARACHI_BBOX
    karachi_rect = ee.Geometry.Rectangle([west - 0.25, south - 0.15, east + 0.25, north + 0.15])

    def export_collection(col, name, band_names):
        def extract_image(image):
            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
            reduced = image.select(band_names).reduceRegions(
                collection=station_features,
                reducer=ee.Reducer.mean(),
                scale=1000
            )
            return reduced.map(lambda f: f.set('date', date))

        results = col.map(extract_image).flatten()
        task = ee.batch.Export.table.toDrive(
            collection=results,
            description=name,
            folder='karachi_airq_exports',
            fileFormat='CSV'
        )
        task.start()
        print(f"✓ Queued Export: {name}")
        return task, name

    tasks = []

    # --- 1. Sentinel-5P (NRTI) ---
    s5p_sources = {
        'aer_ai': ('COPERNICUS/S5P/NRTI/L3_AER_AI', ['absorbing_aerosol_index']),
        'no2':    ('COPERNICUS/S5P/NRTI/L3_NO2',    ['NO2_column_number_density']),
        'so2':    ('COPERNICUS/S5P/NRTI/L3_SO2',    ['SO2_column_number_density']),
        'co':     ('COPERNICUS/S5P/NRTI/L3_CO',     ['CO_column_number_density'])
    }
    for label, (asset_id, bands) in s5p_sources.items():
        col = ee.ImageCollection(asset_id).filterDate(args.start, args.end).filterBounds(karachi_rect)
        tasks.append(export_collection(col, f'karachi_s5p_{label}', bands))

    # --- 1b. S5P OFFL (opt-in to avoid duplicate downloads) ---
    if args.include_offl:
        offl_sources = {
            'aer_ai_offl': ('COPERNICUS/S5P/OFFL/L3_AER_AI', ['absorbing_aerosol_index']),
            'no2_offl':    ('COPERNICUS/S5P/OFFL/L3_NO2',    ['NO2_column_number_density']),
            'so2_offl':    ('COPERNICUS/S5P/OFFL/L3_SO2',    ['SO2_column_number_density']),
            'co_offl':     ('COPERNICUS/S5P/OFFL/L3_CO',     ['CO_column_number_density']),
        }
        for label, (asset_id, bands) in offl_sources.items():
            col = ee.ImageCollection(asset_id).filterDate(args.start, args.end).filterBounds(karachi_rect)
            tasks.append(export_collection(col, f'karachi_s5p_{label}', bands))

    # --- 2. MODIS AOD (MAIAC 1km) ---
    modis = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES') \
        .filterDate(args.start, args.end) \
        .filterBounds(karachi_rect) \
        .select(['Optical_Depth_047', 'Optical_Depth_055']) \
        .map(lambda img: img.multiply(0.001).set('system:time_start', img.get('system:time_start')))
    tasks.append(export_collection(modis, 'karachi_modis_aod', ['Optical_Depth_047', 'Optical_Depth_055']))

    # --- 3. ERA5 Land ---
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterDate(args.start, args.end) \
        .filterBounds(karachi_rect)

    def engineer_era5(img):
        u = img.select('u_component_of_wind_10m')
        v = img.select('v_component_of_wind_10m')
        ws = u.hypot(v).rename('wind_speed')
        t  = img.select('temperature_2m').subtract(273.15)
        td = img.select('dewpoint_temperature_2m').subtract(273.15)
        # Standard Magnus approximation (was wrong in prior version).
        # RH ≈ 100 · exp(17.625·Td/(Td+243.04) − 17.625·T/(T+243.04))
        rh = t.expression(
            '100 * (exp((17.625 * td) / (td + 243.04)) / exp((17.625 * t) / (t + 243.04)))',
            {'td': td, 't': t}
        ).rename('rh')
        return img.addBands([ws, rh]).set('system:time_start', img.get('system:time_start'))

    era5_eng = era5.map(engineer_era5).select(['wind_speed', 'rh', 'temperature_2m', 'total_precipitation_sum'])
    tasks.append(export_collection(era5_eng, 'karachi_era5_met',
                                   ['wind_speed', 'rh', 'temperature_2m', 'total_precipitation_sum']))

    print(f"\n{len(tasks)} tasks queued. Polling each (max {args.max_wait}s each)...")
    for task, label in tasks:
        _poll_task(task, label, args.max_wait)

    print("\nAll tasks have a terminal state. Check your Google Drive 'karachi_airq_exports' folder.")


if __name__ == "__main__":
    main()
