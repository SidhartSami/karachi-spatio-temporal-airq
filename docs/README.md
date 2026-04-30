# 🌍 Spatio-Temporal Air Quality Digital Twin: Karachi (2019–2024)

[![Status](https://img.shields.io/badge/Status-Complete-success)](https://github.com/SidhartSami/karachi-spatio-temporal-airq)
[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A **complete, publication-quality pipeline** for spatio-temporal PM2.5 prediction in Karachi, Pakistan using multi-source satellite data, machine learning, and interactive 3D visualization.

## 📌 Project Overview
This project integrates **high-resolution satellite imagery** (MODIS, Sentinel-5P, ERA5) and **deep learning** (LSTM with Attention) to construct a comprehensive air quality monitoring system. It features:
- **8 executed Jupyter notebooks** with full outputs
- **6 trained ML models** (Random Forest, XGBoost, LSTM, etc.)
- **Interactive 3D Digital Twin** with real-time policy sliders
- **28 output files** (charts, maps, CSV results)

> 🏆 **Best Model:** Random Forest (R²=0.61, RMSE=16.3 µg/m³)  
> 📊 **Dataset:** 14,536 rows × 35 columns (2019–2023)  
> 🗺️ **Coverage:** 8 monitoring stations, 1km² grid resolution

## 🛰️ Data Sources
- **Sentinel-5P (S5P):** Offline (OFFL) products for $NO_2$, $SO_2$, $CO$, and Aerosol Index (AER AI).
- **MODIS (Terra/Aqua):** Aerosol Optical Depth (AOD) at 550nm.
- **ERA5-Land:** Daily meteorological variables (Temperature, Wind Speed/Direction, Relative Humidity, Surface Pressure).
- **VIIRS:** Monthly Nighttime Light (NTL) as an urban activity proxy.
- **Sentinel-2:** NDVI and NDBI (Built-up Index) for land-use context.

## 🛠️ Key Technical Features
- **3-Stage Smart Imputation:**
  - *Short-term:* Linear interpolation (≤7 days).
  - *Medium-term:* Seasonal mean filling (Station × Month).
  - *Long-term:* KNN Imputation using meteorological and spatial features.
- **Advanced Feature Engineering:**
  - **Stagnation Index:** Calculated as $(DTR / Wind Speed)$ to quantify atmospheric trapping.
  - **Pakistan-Specific Flags:** Automated encoding for Ramadan and Eid holidays.
  - **Lag Analysis:** Auto-generation of 1, 3, 7, and 14-day pollutant lags.
- **Outlier Resilience:** 3×IQR Winsorizing to retain extreme pollution events (e.g., industrial spikes) while filtering sensor noise.

## 📂 Repository Structure

### Notebooks (All Executed ✅)
| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `01_data_collection.ipynb` | GEE export orchestration (ERA5, MODIS, S5P, VIIRS, Sentinel-2) | Raw satellite CSVs |
| `02_preprocessing.ipynb` | Data merging, cleaning, row stabilization | `modeling_dataset.csv` |
| `03_eda.ipynb` | Temporal cycles, spatial rankings, correlation analysis | 7 PNG visualizations |
| `04_feature_selection.ipynb` | VIF analysis, LASSO, mutual information, consensus ranking | 6 feature charts |
| `05_models.ipynb` | ML training: RF, XGBoost, LightGBM, SVR, Prophet + SHAP analysis | 5 trained models |
| `06_spatial_analysis.ipynb` | Moran's I, LISA hotspots, IDW interpolation, zone analysis | Spatial maps |
| `07_lstm_digital_twin.ipynb` | LSTM + Attention, multi-horizon forecasting, scenario simulation | `lstm_model.pt` |
| `08_karachi_digital_twin_map.ipynb` | Folium interactive maps, station-level scenarios | HTML maps |

### Key Directories
- `notebooks/data/processed/`: Clean dataset (1.9MB) + feature config
- `notebooks/models/`: 6 trained models (RF 129MB via Git LFS)
- `notebooks/outputs/`: 28 output files (PNG, CSV, HTML)
- `dashboard/`: **13 interactive HTML dashboards** (3D Digital Twin)
- `scratch/`: GEE utility scripts

## 🏆 Model Results (2023 Holdout Test)

| Model | RMSE | MAE | R² | MAPE | Status |
|-------|------|-----|-----|------|--------|
| 🥇 **Random Forest** | **16.30** | **10.63** | **0.612** | **22.0%** | Best Overall |
| 🥈 XGBoost (GPU) | 16.41 | 10.95 | 0.607 | 23.3% | Close Second |
| 🥉 LightGBM | 16.59 | 10.59 | 0.598 | 21.5% | Fastest Training |
| SVR | 17.54 | 11.93 | 0.551 | 25.7% | Baseline |
| Prophet | 40.17 | 29.36 | -1.356 | 51.6% | Poor Fit |
| LSTM | 0.99* | — | 0.99* | — | Training R² (*Multi-horizon forecasting) |

**Key Findings (SHAP Analysis):**
1. `pm25_lag1` (9.5) — Previous day's PM2.5 most predictive
2. `pm25_roll7` (4.5) — 7-day rolling mean
3. `Optical_Depth_055` (2.8) — MODIS AOD
4. `wind_speed` (2.5) — Meteorological driver

**Spatial Analysis:**
- Moran's I = 0.37 (p=0.031) — Significant spatial clustering
- Industrial zones: 39% higher PM2.5 than residential
- WHO guideline exceeded by 11.1× on average

## 🎮 Interactive Dashboard Demo

### 🌐 Open the Digital Twin (No Installation!)
```bash
# Primary: Real-time sliders (1% resolution)
start dashboard/karachi_twin_fully_dynamic.html

# Alternative: Simple model switcher
start dashboard/model_ensemble_average.html
```

### 🎯 Dashboard Features
- **3 Real-Time Sliders:** Industry (0-100%), Traffic (0-100%), Green (0-100%)
- **Live Statistics:** Mean PM2.5, WHO exceedance %, max values
- **3D Visualization:** 357 grid points update instantly as you drag
- **Preset Buttons:** Baseline, Moderate, Aggressive, Extreme policies
- **Model Comparison:** 5 model views with different prediction characteristics

### 📊 Try This
1. Open `karachi_twin_fully_dynamic.html`
2. Drag **Industry** slider from 0→50% — watch SITE/Korangi hotspots drop
3. Add **Green** 40% — see city-wide reduction
4. Check stats: Mean drops from 55→38 µg/m³, WHO exceedance decreases

## 🚀 Getting Started

### Quick Demo (Pre-built)
All notebooks executed, models trained, outputs generated:
```bash
# View interactive dashboard
start dashboard/karachi_twin_fully_dynamic.html

# Or regenerate everything
jupyter notebook notebooks/05_models.ipynb  # Re-run any notebook
```

### Full Reproduction
```bash
# 1. Setup
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Data Collection (requires GEE auth)
python scratch/run_s5p_citywide.py
# Download CSVs from Google Drive → notebooks/data/raw/

# 3. Run Pipeline
python merge_data.py
python phase3_step1_clean_impute.py
python phase3_step3_merge_target.py

# 4. Execute Notebooks (in order)
jupyter notebook notebooks/01_data_collection.ipynb  # Through 08...
```

## 📊 Methodology Highlights
- **Resilient to Missing Data:** 3-stage imputation handles cloud masking
- **Physics-Informed:** Stagnation index captures atmospheric trapping
- **Policy Simulation:** Digital twin tests intervention scenarios
- **Reproducible:** 100% scripted pipeline from satellite download to prediction
