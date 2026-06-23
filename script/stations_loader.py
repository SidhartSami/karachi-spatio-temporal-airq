"""Single source of truth for Karachi monitoring stations.

Import this module instead of hard-coding station coordinates in pipeline
scripts. Station coordinates were previously duplicated across four files
with three different lat/lon conventions (see ISSUES_FOUND.md M2/M3).
"""
import json
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent / "stations.json"
_CONFIG: dict[str, Any] = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

STATIONS_LIST: list[dict[str, Any]] = _CONFIG["stations"]
STATIONS_DICT: dict[str, dict[str, Any]] = {s["name"]: s for s in STATIONS_LIST}
STATION_NAMES: list[str] = [s["name"] for s in STATIONS_LIST]
STATION_NAME_SET = set(STATION_NAMES)

KARACHI_BBOX: list[float] = [
    _CONFIG["_meta"]["bbox_west"],
    _CONFIG["_meta"]["bbox_south"],
    _CONFIG["_meta"]["bbox_east"],
    _CONFIG["_meta"]["bbox_north"],
]

US_CONSULATE_OPENAQ_ID: int = int(_CONFIG["us_consulate_openaq_id"])
GEE_PROJECT_DEFAULT: str = _CONFIG["gee_project_default"]


def to_openaq_latlon(station: dict[str, Any]) -> str:
    """OpenAQ v3 expects 'lat,lon' in the `coordinates` query parameter."""
    return f"{station['lat']},{station['lon']}"


def to_lonlat(station: dict[str, Any]) -> list[float]:
    """Where [lon, lat] is the convention (GEE geometry, ee.Geometry.Point)."""
    return [station["lon"], station["lat"]]


def zone_type(station_name: str) -> str:
    return STATIONS_DICT[station_name]["zone_type"]


def all_industrial() -> list[str]:
    return [s["name"] for s in STATIONS_LIST if s["zone_type"] == "industrial"]


if __name__ == "__main__":
    print(f"Loaded {len(STATIONS_LIST)} stations from {_CONFIG_PATH}")
    for s in STATIONS_LIST:
        print(f"  {s['name']:<22} ({s['lat']:.4f}, {s['lon']:.4f})  [{s['zone_type']}]")
    print(f"\nKARACHI_BBOX        = {KARACHI_BBOX}")
    print(f"US_CONSULATE_ID     = {US_CONSULATE_OPENAQ_ID}")
    print(f"GEE_PROJECT_DEFAULT = {GEE_PROJECT_DEFAULT}")
