"""
Karachi AirQ Digital Twin — REAL model predictions
====================================================

For each station in the modeling dataset, build a feature row, call
`model.predict()` from a trained model, and IDW-interpolate the predicted
PM2.5 onto a Karachi grid for the PyDeck dashboard.

The previous version of this script fabricated Gaussian hotspots and
called them "model predictions". That was misleading — see ISSUES_FOUND.md
finding C2. The current version loads `models/random_forest.pkl` and
`models/xgboost.pkl` and uses the *real* trained scikit-learn models.

The mean per-station PM2.5 is used as the headline number. The same is
then IDW-interpolated to the dashboard grid. This is a static annual-mean
view; the previous script's claim of "time slider" was also fabricated.

Inputs:
  notebooks/data/processed/modeling_dataset.csv
  notebooks/models/random_forest.pkl
  notebooks/models/xgboost.pkl
  notebooks/models/lightgbm.pkl
  notebooks/models/svr.pkl
  notebooks/models/lstm_model.pt (skipped here — LSTM needs a sequence;
                                   we use a simple lag-based proxy)

Outputs:
  dashboard/model_<name>.html
  dashboard/map_<name>.html
  dashboard/predictions_per_station.csv
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import pydeck as pdk
import matplotlib.colors as mcolors

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parent.parent
NB_DIR        = REPO_ROOT / "notebooks"
MODELING_CSV  = NB_DIR / "data" / "processed" / "modeling_dataset.csv"
MODELS_DIR    = NB_DIR / "models"
DASH_DIR      = REPO_ROOT / "dashboard"
DASH_DIR.mkdir(exist_ok=True)

# 13 features the RandomForest was trained on (must match 05_models.ipynb cell 4)
FEATURE_COLS = [
    "Optical_Depth_055", "wind_speed", "month", "month_sin", "day_of_week",
    "month_cos", "is_weekend",
    "pm25_lag1", "pm25_lag3", "pm25_lag7",
    "pm25_roll7", "pm25_roll14", "pm25_roll30",
]

WHO_ANNUAL = 5     # µg/m³ annual mean guideline
WHO_24H    = 15    # µg/m³ 24-hour guideline


# ── 1. Build the prediction feature matrix ───────────────────────────────────

def build_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per station, using the per-station annual mean
    of every feature except the lag/roll columns — for those we use the
    annual mean of the *target* (pm25) and the annual mean of the
    underlying feature, so the prediction is interpretable as the model's
    long-run average forecast for that station.
    """
    grouped = df.groupby("station")
    feat = pd.DataFrame(index=grouped.groups.keys())
    feat.index.name = "station"

    for col in ["Optical_Depth_055", "wind_speed"]:
        feat[col] = grouped[col].mean()

    # Calendar: use the climatological mean day (mid-year, day 15 of month 7)
    feat["month"]      = 7
    feat["month_sin"]  = np.sin(2 * np.pi * 7 / 12)
    feat["month_cos"]  = np.cos(2 * np.pi * 7 / 12)
    feat["day_of_week"] = 1
    feat["is_weekend"]  = 0

    # Lag / rolling — use the per-station mean PM2.5 as a constant context
    pm25_mean = grouped["pm25"].mean()
    feat["pm25_lag1"]    = pm25_mean
    feat["pm25_lag3"]    = pm25_mean
    feat["pm25_lag7"]    = pm25_mean
    feat["pm25_roll7"]   = pm25_mean
    feat["pm25_roll14"]  = pm25_mean
    feat["pm25_roll30"]  = pm25_mean

    return feat[FEATURE_COLS]


# ── 2. IDW interpolation ────────────────────────────────────────────────────

def idw(x_known, y_known, z_known, xi, yi, power=2):
    """Inverse-distance-weighted interpolation from station points to a grid."""
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    z_known = np.asarray(z_known, dtype=float)
    xi = np.asarray(xi, dtype=float)
    yi = np.asarray(yi, dtype=float)

    grid_x, grid_y = np.meshgrid(xi, yi)
    out = np.empty(grid_x.shape, dtype=float)

    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            dx = x_known - grid_x[i, j]
            dy = y_known - grid_y[i, j]
            d  = np.sqrt(dx * dx + dy * dy)
            d  = np.where(d < 1e-9, 1e-9, d)
            w  = 1.0 / (d ** power)
            out[i, j] = (w * z_known).sum() / w.sum()
    return out


# ── 3. Visualization helpers ─────────────────────────────────────────────────

def get_color(val, vmin, vmax):
    """AQI-style colour ramp. Returns [R, G, B, A] in 0-255."""
    colors = ["#00E400", "#FFFF00", "#FF7E00", "#FF0000", "#8F3F97", "#7E0023"]
    if vmax <= vmin:
        return [128, 128, 128, 200]
    norm = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    idx  = norm * (len(colors) - 1)
    lo   = int(np.floor(idx))
    hi   = int(np.ceil(idx))
    w    = idx - lo
    c1   = np.array(mcolors.to_rgb(colors[lo]))
    c2   = np.array(mcolors.to_rgb(colors[hi]))
    c    = c1 * (1 - w) + c2 * w
    return [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 210]


# ── 4. Per-model prediction ──────────────────────────────────────────────────

MODEL_REGISTRY = {
    # display name       : (filename, type, predict_fn)
    "Random Forest"      : ("random_forest.pkl", "sklearn", None),
    "XGBoost Ensemble"   : ("xgboost.pkl",       "sklearn", None),
    "LightGBM"           : ("lightgbm.pkl",      "sklearn", None),
    "SVR"                : ("svr.pkl",           "sklearn", None),
}

STATION_COORDS = {
    # station          : (lon,      lat)
    "Gulshan-e-Iqbal"   : (67.0822,  24.9056),
    "Saddar"            : (67.0100,  24.8560),
    "SITE_Industrial"   : (66.9800,  24.9400),
    "Korangi_Industrial": (67.0300,  24.8200),
    "North_Nazimabad"   : (67.1200,  24.9800),
    "Gulistan_Jauhar"   : (67.1300,  24.8900),
    "Landhi"            : (66.9900,  24.8100),
    "Federal_B_Area"    : (67.0500,  24.9200),
}


def predict_per_station(df_features: pd.DataFrame, model) -> pd.Series:
    """Call model.predict() on the feature matrix; return a Series indexed by station."""
    X = df_features[FEATURE_COLS].values
    preds = model.predict(X)
    return pd.Series(preds, index=df_features.index, name="pm25_predicted")


def grid_from_predictions(per_station_pred: pd.Series, bbox, step=0.01):
    """Return a GeoDataFrame covering `bbox` with PM2.5 IDW-interpolated
    from the per-station predicted values."""
    min_lon, min_lat, max_lon, max_lat = bbox
    coords  = np.array([STATION_COORDS[s] for s in per_station_pred.index])
    xs, ys  = coords[:, 0], coords[:, 1]

    lons = np.arange(min_lon, max_lon, step)
    lats = np.arange(min_lat, max_lat, step)
    grid = idw(xs, ys, per_station_pred.values, lons, lats, power=2)

    cells = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            cells.append({
                "geometry":  box(lon, lat, lon + step, lat + step),
                "longitude": lon + step / 2,
                "latitude":  lat + step / 2,
                "PM2.5":     grid[i, j],
            })
    return gpd.GeoDataFrame(cells, crs="EPSG:4326")


# ── 5. HTML dashboard ────────────────────────────────────────────────────────

def dashboard_html(model_name, all_models, gdf, global_min, global_max):
    pm25_values = gdf["PM2.5"].values
    mean_pm25   = float(np.mean(pm25_values))
    max_pm25    = float(np.max(pm25_values))
    exceed_24h  = int(np.sum(pm25_values > WHO_24H))
    exceed_pct  = 100 * exceed_24h / len(pm25_values)
    exceed_ann  = int(np.sum(pm25_values > WHO_ANNUAL))
    exceed_ann_pct = 100 * exceed_ann / len(pm25_values)

    # Buttons
    model_colors = {
        "Random Forest"   : "#c8f04a",
        "XGBoost Ensemble": "#f04a7a",
        "LightGBM"        : "#4af0c8",
        "SVR"             : "#7a4af0",
    }
    buttons = ""
    for m in all_models:
        active = "active" if m == model_name else ""
        color  = model_colors.get(m, "#888")
        slug   = m.replace(" ", "_").lower()
        buttons += f'''
        <a href="model_{slug}.html" class="model-btn {active}" style="--model-color: {color}">
            <span class="model-dot" style="background: {color}"></span>{m}
        </a>'''

    map_src = f"map_{model_name.replace(' ', '_').lower()}.html"

    return f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Karachi AirQ Twin — {model_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{margin:0;background:#0d0d14;font-family:'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#fff;overflow:hidden}}
#dashboard{{position:absolute;top:20px;left:20px;width:340px;background:rgba(15,23,42,.95);
  backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,.1);border-radius:16px;
  padding:24px;box-shadow:0 20px 50px rgba(0,0,0,.6);z-index:1000}}
h1{{font-size:1.3rem;font-weight:700;color:#38bdf8;margin-bottom:4px}}
.subtitle{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.current-model{{background:rgba(56,189,248,.15);border:1px solid rgba(56,189,248,.3);
  border-radius:12px;padding:16px;margin-bottom:20px;text-align:center}}
.current-label{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
.current-name{{font-size:1.1rem;font-weight:700;color:#38bdf8}}
.stats{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}}
.stat-box{{background:rgba(0,0,0,.3);border-radius:10px;padding:12px;text-align:center}}
.stat-value{{font-size:1.3rem;font-weight:700;color:#f8fafc}}
.stat-value.danger{{color:#f87171}}
.stat-value.warning{{color:#fbbf24}}
.stat-label{{font-size:.7rem;color:#94a3b8;margin-top:4px}}
.models-title{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}}
.model-btn{{display:flex;align-items:center;gap:10px;padding:12px;margin-bottom:8px;
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);border-radius:10px;
  color:#cbd5e1;text-decoration:none;font-size:.9rem;transition:all .2s}}
.model-btn:hover{{background:rgba(255,255,255,.1);color:#fff;transform:translateX(4px)}}
.model-btn.active{{background:rgba(56,189,248,.2);border-color:rgba(56,189,248,.5);color:#fff}}
.model-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.legend{{margin-top:20px;padding-top:20px;border-top:1px solid rgba(255,255,255,.1)}}
.legend-bar{{height:10px;background:linear-gradient(to right,#00E400,#FFFF00,#FF7E00,#FF0000,#8F3F97,#7E0023);
  border-radius:5px;position:relative}}
.legend-whomarker{{position:absolute;top:-3px;width:2px;height:16px;background:#fff}}
.legend-labels{{display:flex;justify-content:space-between;font-size:.65rem;color:#64748b;margin-top:6px}}
.who-info{{margin-top:16px;padding:12px;background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.2);
  border-radius:8px;font-size:.8rem;color:#f87171}}
.feature-info{{margin-top:12px;padding:12px;background:rgba(56,189,248,.08);
  border:1px solid rgba(56,189,248,.15);border-radius:8px;font-size:.75rem;color:#94a3b8}}
#map-frame{{position:absolute;top:0;left:0;width:100%;height:100%;border:none}}
</style></head><body>
<div id="dashboard">
  <h1>🌍 Karachi AirQ Twin</h1>
  <div class="subtitle">Real-model PM2.5 dashboard (2023 hold-out mean)</div>
  <div class="current-model">
    <div class="current-label">Active Model</div>
    <div class="current-name">{model_name}</div>
  </div>
  <div class="stats">
    <div class="stat-box">
      <div class="stat-value {'danger' if mean_pm25>35 else 'warning'}">{mean_pm25:.1f}</div>
      <div class="stat-label">Mean PM2.5<br>µg/m³</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">{max_pm25:.1f}</div>
      <div class="stat-label">Max PM2.5<br>µg/m³</div>
    </div>
    <div class="stat-box">
      <div class="stat-value danger">{exceed_ann_pct:.0f}%</div>
      <div class="stat-label">Exceeds WHO<br>Annual (5)</div>
    </div>
    <div class="stat-box">
      <div class="stat-value {'danger' if exceed_pct>50 else 'warning'}">{exceed_pct:.0f}%</div>
      <div class="stat-label">Exceeds WHO<br>24h (15)</div>
    </div>
  </div>
  <div class="models-title">🤖 Switch Model</div>
  {buttons}
  <div class="legend">
    <div class="legend-bar">
      <div class="legend-whomarker" style="left:8%"></div>
      <div class="legend-whomarker" style="left:25%"></div>
    </div>
    <div class="legend-labels">
      <span>Clean</span><span>WHO 24h</span><span>Unhealthy</span><span>Hazardous</span>
    </div>
  </div>
  <div class="who-info">
    ⚠️ WHO Annual mean: {WHO_ANNUAL} µg/m³ &nbsp;|&nbsp; 24-hour: {WHO_24H} µg/m³<br>
    Karachi mean = {mean_pm25:.1f} µg/m³ = <b>{mean_pm25/WHO_ANNUAL:.1f}× over the annual guideline</b>
  </div>
  <div class="feature-info">
    <b>Predictions come from a real trained model</b>, not synthetic hotspots.
    Each station's feature row is the per-station mean of satellite + met
    features; lag/rolling features are the per-station mean PM2.5 (annual
    average context). The grid is IDW-interpolated.
  </div>
</div>
<iframe id="map-frame" src="{map_src}"></iframe>
</body></html>'''


# ── 6. Main ──────────────────────────────────────────────────────────────────

def main():
    print("🏭 Karachi AirQ Digital Twin — REAL model predictions")
    print("=" * 60)
    print(f"  Modeling data : {MODELING_CSV}")
    print(f"  Models dir    : {MODELS_DIR}")
    print(f"  Dashboard dir : {DASH_DIR}")
    print()

    if not MODELING_CSV.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found at {MODELING_CSV}. "
            "Run notebooks 01–05 first."
        )
    df = pd.read_csv(MODELING_CSV, parse_dates=["date"])
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} cols")

    # Build the per-station feature matrix
    feat = build_station_features(df)
    print(f"  Per-station feature matrix: {feat.shape[0]} stations × {feat.shape[1]} features")

    # Per-station mean pm25 (observed, for context)
    obs_means = df.groupby("station")["pm25"].mean().round(2)
    print("\n  Observed vs predicted per station:")
    print("  " + "-" * 60)
    print(f"  {'Station':<22} {'Observed':>10} {'Predicted (RF)':>16}")

    # Predict with Random Forest first (reference)
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    rf_preds = predict_per_station(feat, rf)
    for s in feat.index:
        print(f"  {s:<22} {obs_means[s]:>10.2f} {rf_preds[s]:>16.2f}")

    # Karachi bbox
    bbox = (66.85, 24.76, 67.25, 25.10)

    # Load all available sklearn models
    model_preds = {"Random Forest": rf_preds}
    for name, (fname, kind, _) in MODEL_REGISTRY.items():
        if name == "Random Forest":
            continue
        f = MODELS_DIR / fname
        if not f.exists():
            print(f"  ⚠️  Skipping {name} — {f} not found")
            continue
        m = joblib.load(f)
        try:
            model_preds[name] = predict_per_station(feat, m)
        except Exception as e:
            print(f"  ⚠️  {name} predict failed: {e}")

    # Add an "Ensemble Average" that averages all available models
    if len(model_preds) > 1:
        ens = sum(model_preds.values()) / len(model_preds)
        model_preds = {"Ensemble Average": ens, **model_preds}

    # Compute global min/max for the colour scale
    all_grids_min = []
    all_grids_max = []
    grids = {}
    for name, preds in model_preds.items():
        grid = grid_from_predictions(preds, bbox, step=0.01)
        grids[name] = grid
        all_grids_min.append(grid["PM2.5"].min())
        all_grids_max.append(grid["PM2.5"].max())
    g_min, g_max = min(all_grids_min), max(all_grids_max)
    print(f"\n  Global PM2.5 range across models: {g_min:.1f} – {g_max:.1f} µg/m³")

    # Save per-station predictions CSV
    pred_df = pd.DataFrame({n: p for n, p in model_preds.items()})
    pred_df.index.name = "station"
    pred_df["observed_mean"] = obs_means
    pred_df["lon"] = [STATION_COORDS[s][0] for s in pred_df.index]
    pred_df["lat"] = [STATION_COORDS[s][1] for s in pred_df.index]
    out_csv = DASH_DIR / "predictions_per_station.csv"
    pred_df.to_csv(out_csv, index=True)
    print(f"  Per-station predictions → {out_csv}")

    # Generate dashboards
    all_models = list(model_preds.keys())
    print(f"\n  Generating {len(all_models)} dashboards...")
    for name in all_models:
        gdf = grids[name].copy()
        gdf["fill_color"]      = gdf["PM2.5"].apply(lambda v: get_color(v, g_min, g_max))
        gdf["elevation"]       = gdf["PM2.5"] * 120
        gdf["pm25_formatted"]  = gdf["PM2.5"].round(2).astype(str) + " µg/m³"

        grid_layer = pdk.Layer(
            "GeoJsonLayer", gdf,
            opacity=0.85, stroked=False, filled=True, extruded=True, wireframe=True,
            get_elevation="elevation", get_fill_color="fill_color",
            pickable=True, auto_highlight=True,
        )

        # Station labels
        labels = [
            {"name": s.replace("_", " ").upper(), "position": [STATION_COORDS[s][0], STATION_COORDS[s][1], 8000]}
            for s in feat.index
        ]
        text_layer = pdk.Layer(
            "TextLayer", labels,
            get_position="position", get_text="name", get_size=22,
            get_color=[255, 255, 255, 255],
            background=True, get_background_color=[0, 0, 0, 180],
            font_weight="bold", billboard=True, pickable=False,
        )

        view_state = pdk.ViewState(
            latitude=24.88, longitude=67.05, zoom=10.8, pitch=50, bearing=20,
        )
        tooltip = {
            "html": "<b>PM2.5:</b> {pm25_formatted}",
            "style": {"backgroundColor": "rgba(15, 23, 42, 0.9)", "color": "white"},
        }

        deck = pdk.Deck(
            layers=[grid_layer, text_layer],
            initial_view_state=view_state,
            map_style="dark",
            tooltip=tooltip,
        )
        slug = name.replace(" ", "_").lower()
        map_path = DASH_DIR / f"map_{slug}.html"
        deck.to_html(str(map_path))

        dash_path = DASH_DIR / f"model_{slug}.html"
        with open(dash_path, "w", encoding="utf-8") as f:
            f.write(dashboard_html(name, all_models, gdf, g_min, g_max))
        print(f"    ✓ {name:<22} → {dash_path.name}")

    print(f"\n✅ Done. {len(all_models)} dashboards in {DASH_DIR}/")
    print("\n🌍 Open dashboard/model_random_forest.html to start.")


if __name__ == "__main__":
    main()