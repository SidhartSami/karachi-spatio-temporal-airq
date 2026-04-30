"""
Phase 3 — Step 2: Fetch Ground-Truth PM2.5 from OpenAQ v3
==========================================================
What this script does:
  1. Queries the OpenAQ v3 API for stations near each of Karachi's 8
     monitoring locations using a radius search.
  2. Downloads daily PM2.5 measurements for 2019–2024.
  3. Aggregates to a daily mean per station.
  4. Saves: data/raw/openaq_pm25_karachi.csv

Strategy:
  - OpenAQ v3 uses sensor-location IDs. We first discover nearby sensors
    for each of our 8 named stations via /v3/locations?coordinates=...
  - We then pull hourly/daily measurements and resample to daily mean.
  - If no sensor is found within RADIUS_KM, we expand to FALLBACK_RADIUS_KM.
  - The US Consulate Karachi sensor is hardcoded as a high-quality anchor.

Usage:
  python phase3_step2_fetch_groundtruth.py \
      --output data/raw/openaq_pm25_karachi.csv \
      [--api-key YOUR_OPENAQ_KEY]     # Optional but avoids rate limits
      [--start 2019-01-01]
      [--end   2024-12-31]

Get a free API key at: https://explore.openaq.org/register

Dependencies:
  pip install requests pandas tqdm python-dateutil
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Station Registry ──────────────────────────────────────────────────────────
# Your 8 GEE extraction stations with their exact coordinates.
STATIONS = [
    {"name": "Gulshan-e-Iqbal",   "lat": 24.9056, "lon": 67.0822},
    {"name": "Saddar",            "lat": 24.8560, "lon": 67.0100},
    {"name": "SITE_Industrial",   "lat": 24.9400, "lon": 66.9800},
    {"name": "Korangi_Industrial","lat": 24.8200, "lon": 67.0300},
    {"name": "North_Nazimabad",   "lat": 24.9800, "lon": 67.1200},
    {"name": "Gulistan_Jauhar",   "lat": 24.8900, "lon": 67.1300},
    {"name": "Landhi",            "lat": 24.8100, "lon": 66.9900},
    {"name": "Federal_B_Area",    "lat": 24.9200, "lon": 67.0500},
]

# US Consulate Karachi — known high-quality PM2.5 sensor, hardcoded as anchor.
# OpenAQ location ID (verify at https://explore.openaq.org)
US_CONSULATE_OPENAQ_ID = 225442   # Update if this changes

OPENAQ_BASE        = "https://api.openaq.org/v3"
RADIUS_KM          = 3      # First search radius per station
FALLBACK_RADIUS_KM = 8      # If nothing found, expand
RATE_LIMIT_SLEEP   = 0.4    # seconds between API calls (be a good citizen)
MAX_RETRIES        = 3


# ── API Helpers ───────────────────────────────────────────────────────────────

def build_headers(api_key: str | None) -> dict:
    h = {"Accept": "application/json"}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def safe_get(url: str, params: dict, headers: dict, retries: int = MAX_RETRIES) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                log.warning("Rate limited — sleeping %ds", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            time.sleep(RATE_LIMIT_SLEEP)
            return r.json()
        except requests.RequestException as e:
            log.warning("Attempt %d/%d failed: %s", attempt + 1, retries, e)
            time.sleep(2 ** attempt)
    log.error("All retries exhausted for %s", url)
    return None


# ── Station Discovery ─────────────────────────────────────────────────────────

def find_location_ids(station: dict, headers: dict, radius_km: int) -> list[int]:
    """Find OpenAQ location IDs near the given station coordinates."""
    params = {
        "coordinates": f"{station['lat']},{station['lon']}",
        "radius":       radius_km * 1000,   # API expects metres
        "parameters":   "pm25",
        "limit":        10,
    }
    data = safe_get(f"{OPENAQ_BASE}/locations", params, headers)
    if not data or not data.get("results"):
        return []
    ids = [r["id"] for r in data["results"]]
    log.info(
        "  Found %d sensor(s) within %dkm of %-22s → IDs: %s",
        len(ids), radius_km, station["name"], ids
    )
    return ids


# ── Measurement Fetching ──────────────────────────────────────────────────────

def fetch_measurements_for_location(
    location_id: int,
    start: str,
    end: str,
    headers: dict,
) -> pd.DataFrame:
    """
    Fetch all PM2.5 measurements for a location between start and end dates.
    OpenAQ v3 returns paginated results; we handle all pages.
    """
    all_rows = []
    page     = 1
    limit    = 1000

    while True:
        params = {
            "location_id":  location_id,
            "parameter":    "pm25",
            "date_from":    f"{start}T00:00:00Z",
            "date_to":      f"{end}T23:59:59Z",
            "limit":        limit,
            "page":         page,
        }
        data = safe_get(f"{OPENAQ_BASE}/measurements", params, headers)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            all_rows.append({
                "datetime": r.get("period", {}).get("datetimeFrom", {}).get("utc"),
                "value":    r.get("value"),
            })

        meta  = data.get("meta", {})
        total = meta.get("found", 0)
        if page * limit >= total:
            break
        page += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"] >= 0]   # Remove negative sensor errors
    return df


def aggregate_to_daily(df: pd.DataFrame, station_name: str, location_id: int) -> pd.DataFrame:
    """Resample hourly measurements to a daily mean."""
    if df.empty:
        return pd.DataFrame()
    df["date"] = df["datetime"].dt.date.astype(str)
    daily = (
        df.groupby("date")["value"]
        .agg(pm25_mean="mean", pm25_count="count")
        .reset_index()
    )
    daily["station"]     = station_name
    daily["location_id"] = location_id
    return daily


# ── Fallback: City-wide daily from US Consulate ───────────────────────────────

def fetch_us_consulate_anchor(start: str, end: str, headers: dict) -> pd.DataFrame:
    """
    The US Consulate Karachi sensor is a single fixed point but is the most
    reliable long-running PM2.5 monitor in the city. We fetch it separately
    and it can serve as a city-wide anchor / validation reference.
    """
    log.info("Fetching US Consulate Karachi anchor sensor (ID=%d)…", US_CONSULATE_OPENAQ_ID)
    raw = fetch_measurements_for_location(US_CONSULATE_OPENAQ_ID, start, end, headers)
    if raw.empty:
        log.warning("No data returned for US Consulate sensor.")
        return pd.DataFrame()
    daily = aggregate_to_daily(raw, "US_Consulate_Karachi", US_CONSULATE_OPENAQ_ID)
    log.info("  US Consulate: %d daily records", len(daily))
    return daily


# ── Main Orchestration ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",  required=True,
                        help="Output CSV path (e.g. data/raw/openaq_pm25_karachi.csv)")
    parser.add_argument("--api-key", default=None,
                        help="OpenAQ v3 API key (optional but recommended)")
    parser.add_argument("--start",   default="2019-01-01")
    parser.add_argument("--end",     default="2024-12-31")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers     = build_headers(args.api_key)

    all_daily = []

    # ── US Consulate anchor (always fetch) ────────────────────────────────────
    anchor = fetch_us_consulate_anchor(args.start, args.end, headers)
    if not anchor.empty:
        all_daily.append(anchor)

    # ── Per-station discovery + fetch ─────────────────────────────────────────
    log.info("Discovering sensors for %d stations…", len(STATIONS))
    for station in tqdm(STATIONS, desc="Stations"):
        log.info("Station: %s", station["name"])
        ids = find_location_ids(station, headers, RADIUS_KM)

        if not ids:
            log.warning(
                "  No sensors within %dkm — trying fallback %dkm radius…",
                RADIUS_KM, FALLBACK_RADIUS_KM
            )
            ids = find_location_ids(station, headers, FALLBACK_RADIUS_KM)

        if not ids:
            log.warning("  ⚠ No OpenAQ sensors found for %s — skipping.", station["name"])
            continue

        # Use the closest sensor (first result from OpenAQ, which sorts by distance)
        chosen_id = ids[0]
        log.info("  Using location_id=%d for %s", chosen_id, station["name"])

        raw = fetch_measurements_for_location(chosen_id, args.start, args.end, headers)
        if raw.empty:
            log.warning("  ⚠ No measurements returned for %s.", station["name"])
            continue

        daily = aggregate_to_daily(raw, station["name"], chosen_id)
        log.info("  %s → %d daily records", station["name"], len(daily))
        all_daily.append(daily)

    if not all_daily:
        log.error("No data collected. Check your API key and network connection.")
        return

    # ── Combine & save ────────────────────────────────────────────────────────
    df_final = pd.concat(all_daily, ignore_index=True)
    df_final = df_final.sort_values(["station", "date"]).reset_index(drop=True)

    # Quality filter: require at least 6 hourly readings to trust a daily mean
    before = len(df_final)
    df_final = df_final[df_final["pm25_count"] >= 6]
    log.info("Quality filter: kept %d/%d daily records (≥6 hourly readings)", len(df_final), before)

    df_final.to_csv(output_path, index=False)
    log.info("─" * 60)
    log.info("Ground truth saved → %s", output_path)
    log.info("Shape : %s", df_final.shape)
    log.info("Stations collected : %s", df_final["station"].unique().tolist())
    log.info("Date range         : %s → %s",
             df_final["date"].min(), df_final["date"].max())

    missing_pct = (1 - len(df_final) / (len(STATIONS) * 365 * 6)) * 100
    log.info("Estimated coverage : %.1f%% missing (vs perfect 6-yr record)", missing_pct)
    log.info("─" * 60)
    log.info("⚠  IMPORTANT: OpenAQ coverage for Karachi is sparse.")
    log.info("   If many stations returned 0 records, proceed to Step 2b:")
    log.info("   Use the US Consulate anchor as a city-wide target variable,")
    log.info("   or run the spatial interpolation fallback in Step 3.")


if __name__ == "__main__":
    main()
