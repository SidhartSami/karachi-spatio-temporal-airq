"""Phase 3 — Step 2: Fetch Ground-Truth PM2.5 from OpenAQ v3.

Fixes vs prior version (ISSUES_FOUND.md):
  M5  — hardcoded US_CONSULATE_OPENAQ_ID = 225442; now reads from
        $OPENAQ_US_CONSULATE_ID with default 225442 and a startup warning.
  M14 — coverage filter relaxed from pm25_count >= 6 to >= 3 hourly readings
        (Karachi OpenAQ historically sparse). Coverage denominator now
        computed from actual queried date range, not inflated 6-year estimate.
  New: every output row carries an explicit `pm25_source` column
        (`openaq_exact` or `openaq_us_consulate`).
  OpenAQ v3 API path corrected: the v3 API does NOT accept
        `GET /v3/measurements?location_id=…` (404). The correct flow is:
        1) /v3/locations  (find by coords, returns location_ids)
        2) /v3/locations/{id}/sensors  (list sensors for that location)
        3) /v3/sensors/{sensor_id}/measurements  (paginated hourly readings)

Behavior when credentials missing: the script does NOT fail. It logs a warning
that anonymous tier is rate-limited (~10 req/min) and proceeds. The fallback
chain (in rebuild_master_dataset.py) will catch sparse coverage.

Usage:
  python phase3_step2_fetch_groundtruth.py \\
      --output data/raw/openaq_pm25_karachi.csv \\
      [--api-key YOUR_OPENAQ_KEY]
  export OPENAQ_KEY=...              # alternative to --api-key
  export OPENAQ_US_CONSULATE_ID=...  # override the default 225442
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from stations_loader import STATIONS_LIST, US_CONSULATE_OPENAQ_ID, to_openaq_latlon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OPENAQ_BASE = "https://api.openaq.org/v3"
RADIUS_KM = 3
FALLBACK_RADIUS_KM = 8
RATE_LIMIT_SLEEP = 0.4
MAX_RETRIES = 3
MIN_HOURLY_READINGS = 3   # was 6; Karachi coverage is sparse


def _resolve_consulate_id() -> int:
    """Read $OPENAQ_US_CONSULATE_ID; warn loudly if user has not overridden
    the default (the ID may have changed)."""
    env_id = os.environ.get("OPENAQ_US_CONSULATE_ID")
    if env_id and env_id.isdigit():
        return int(env_id)
    if env_id is None:
        log.warning(
            "OPENAQ_US_CONSULATE_ID not set — using default %d. Verify at "
            "https://openaq.org/location/%d that this sensor is still active.",
            US_CONSULATE_OPENAQ_ID, US_CONSULATE_OPENAQ_ID,
        )
    return US_CONSULATE_OPENAQ_ID


def build_headers(api_key):
    h = {"Accept": "application/json"}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def safe_get(url, params, headers, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("Rate limited — sleeping %ds", wait)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                log.error("404 Not Found: %s params=%s", url, params)
                return None
            r.raise_for_status()
            time.sleep(RATE_LIMIT_SLEEP)
            return r.json()
        except requests.RequestException as e:
            log.warning("Attempt %d/%d failed: %s", attempt + 1, retries, e)
            time.sleep(2 ** attempt)
    log.error("All retries exhausted for %s", url)
    return None


def find_location_ids(station, headers, radius_km):
    """Find OpenAQ location IDs near the given station coordinates."""
    params = {
        "coordinates": to_openaq_latlon(station),
        "radius":       radius_km * 1000,
        "parameters":   "pm25",
        "limit":        10,
    }
    data = safe_get(f"{OPENAQ_BASE}/locations", params, headers)
    if not data or not data.get("results"):
        return []
    ids = [r["id"] for r in data["results"]]
    log.info("  Found %d location(s) within %dkm of %-22s → IDs: %s",
             len(ids), radius_km, station["name"], ids)
    return ids


def find_pm25_sensor_for_location(location_id: int, headers):
    """Return the PM2.5 sensor id attached to a location, or None if none."""
    data = safe_get(f"{OPENAQ_BASE}/locations/{location_id}/sensors",
                    {}, headers)
    if not data or not data.get("results"):
        return None
    for s in data["results"]:
        p = s.get("parameter", {}) or {}
        if p.get("name") == "pm25":
            return s["id"]
    return None


def fetch_measurements_for_sensor(sensor_id: int, start: str, end: str,
                                   headers) -> pd.DataFrame:
    """Fetch all PM2.5 measurements for a sensor between start and end dates.

    Uses the v3 endpoint /v3/sensors/{id}/measurements, paginated."""
    all_rows = []
    page = 1
    limit = 1000

    while True:
        params = {
            "date_from": f"{start}T00:00:00Z",
            "date_to":   f"{end}T23:59:59Z",
            "limit":     limit,
            "page":      page,
        }
        data = safe_get(f"{OPENAQ_BASE}/sensors/{sensor_id}/measurements",
                        params, headers)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            period = r.get("period") or {}
            dt_from = (period.get("datetimeFrom") or {}).get("utc")
            all_rows.append({
                "datetime": dt_from,
                "value":    r.get("value"),
            })

        meta = data.get("meta", {}) or {}
        found = meta.get("found")
        if isinstance(found, str) and found.startswith(">"):
            page += 1
        elif isinstance(found, int) and page * limit >= found:
            break
        elif not found:
            break
        else:
            page += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"] >= 0]
    return df


def aggregate_to_daily(df, station_name, sensor_id, source):
    if df.empty:
        return pd.DataFrame()
    df["date"] = df["datetime"].dt.date.astype(str)
    daily = (
        df.groupby("date")["value"]
        .agg(pm25_mean="mean", pm25_count="count")
        .reset_index()
    )
    daily["station"]     = station_name
    daily["sensor_id"]   = sensor_id
    daily["pm25_source"] = source
    return daily


def fetch_us_consulate_anchor(start, end, headers):
    consulate_id = _resolve_consulate_id()
    log.info("Fetching US Consulate Karachi anchor (location %d)…", consulate_id)
    sensor_id = find_pm25_sensor_for_location(consulate_id, headers)
    if not sensor_id:
        log.warning("No PM2.5 sensor for US Consulate location %d.", consulate_id)
        return pd.DataFrame()
    raw = fetch_measurements_for_sensor(sensor_id, start, end, headers)
    if raw.empty:
        log.warning("No PM2.5 data from US Consulate sensor %d.", sensor_id)
        return pd.DataFrame()
    daily = aggregate_to_daily(raw, "US_Consulate_Karachi", sensor_id,
                               source="openaq_us_consulate")
    log.info("  US Consulate: %d daily records (sensor %d)",
             len(daily), sensor_id)
    return daily


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",  required=True,
                        help="Output CSV path (e.g. data/raw/openaq_pm25_karachi.csv)")
    parser.add_argument("--api-key", default=None,
                        help="OpenAQ v3 API key (or set $OPENAQ_KEY).")
    parser.add_argument("--start",   default="2019-01-01")
    parser.add_argument("--end",     default="2024-12-31")
    parser.add_argument("--min-readings", type=int, default=MIN_HOURLY_READINGS,
                        help=f"Min hourly readings to trust a daily mean "
                             f"(default: {MIN_HOURLY_READINGS}).")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAQ_KEY")
    if not api_key:
        log.warning(
            "OPENAQ_KEY not set and --api-key not given. Proceeding with "
            "anonymous tier (rate-limited to ~10 req/min). Register at "
            "https://explore.openaq.org/register for ~10× faster fetch."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = build_headers(api_key)

    all_daily = []

    # ── US Consulate anchor (always fetch) ────────────────────────────────────
    anchor = fetch_us_consulate_anchor(args.start, args.end, headers)
    if not anchor.empty:
        all_daily.append(anchor)

    # ── Per-station discovery + fetch ─────────────────────────────────────────
    log.info("Discovering sensors for %d stations…", len(STATIONS_LIST))
    for station in tqdm(STATIONS_LIST, desc="Stations"):
        log.info("Station: %s", station["name"])
        location_ids = find_location_ids(station, headers, RADIUS_KM)

        if not location_ids:
            log.warning("  No locations within %dkm — trying %dkm radius…",
                        RADIUS_KM, FALLBACK_RADIUS_KM)
            location_ids = find_location_ids(station, headers, FALLBACK_RADIUS_KM)

        if not location_ids:
            log.warning("  ⚠ No OpenAQ locations found for %s — skipping.",
                        station["name"])
            continue

        # Try each location; use the first one with a PM2.5 sensor that returns data
        daily = pd.DataFrame()
        chosen = None
        for loc_id in location_ids:
            sensor_id = find_pm25_sensor_for_location(loc_id, headers)
            if not sensor_id:
                continue
            raw = fetch_measurements_for_sensor(sensor_id, args.start, args.end, headers)
            if raw.empty:
                continue
            daily = aggregate_to_daily(raw, station["name"], sensor_id,
                                       source="openaq_exact")
            chosen = (loc_id, sensor_id, len(daily))
            break

        if chosen is None or daily.empty:
            log.warning("  ⚠ No PM2.5 measurements for any nearby location.")
            continue

        log.info("  %s → %d daily records (location %d, sensor %d)",
                 station["name"], chosen[2], chosen[0], chosen[1])
        all_daily.append(daily)

    if not all_daily:
        log.error("No data collected. Check your API key and network connection.")
        sys.exit(1)

    df_final = pd.concat(all_daily, ignore_index=True)
    df_final = df_final.sort_values(["station", "date"]).reset_index(drop=True)

    # Quality filter: require >= N hourly readings (configurable)
    before = len(df_final)
    df_final = df_final[df_final["pm25_count"] >= args.min_readings]
    log.info("Quality filter (>=%d hourly readings): kept %d/%d daily records",
             args.min_readings, len(df_final), before)

    df_final.to_csv(output_path, index=False)
    log.info("─" * 60)
    log.info("Ground truth saved → %s", output_path)
    log.info("Shape : %s", df_final.shape)
    log.info("Stations collected : %s", df_final["station"].unique().tolist())
    log.info("Date range         : %s → %s",
             df_final["date"].min(), df_final["date"].max())
    log.info("pm25_source distribution:")
    for src, n in df_final["pm25_source"].value_counts().items():
        log.info("  %-25s %5d", src, n)

    # Honest coverage report (denominator = actual queried range × stations)
    actual_start = pd.Timestamp(args.start)
    actual_end   = pd.Timestamp(args.end)
    actual_days  = (actual_end - actual_start).days + 1
    expected_rows = len(STATIONS_LIST) * actual_days
    missing_pct = (1 - len(df_final) / max(expected_rows, 1)) * 100
    log.info("─" * 60)
    log.info("Coverage: %d daily records vs %d expected "
             "(stations=%d, days=%d). Missing=%.1f%%",
             len(df_final), expected_rows, len(STATIONS_LIST),
             actual_days, missing_pct)
    log.info("─" * 60)
    log.info("⚠  IMPORTANT: OpenAQ coverage for Karachi is historically sparse.")
    log.info("   If many stations returned 0 records, the orchestrator will "
             "fall back to the US Consulate anchor and then to MERRA-2 "
             "citywide (script/phase3_step2_gee_pm25.py).")


if __name__ == "__main__":
    main()
