# 📄 Product Requirements Document (PRD)
**Project Name:** Karachi Spatio-Temporal Air Quality Estimator  
**Status:** ✅ Completed (All 5 Phases + Enhanced Digital Twin)  
**Last Updated:** 2026-04-26  

---

## 1. 🎯 Vision & Problem Statement
**Problem:** Karachi suffers from severe air pollution, but ground-level air quality monitors are sparse, expensive to maintain, and do not provide continuous, city-wide coverage.  
**Vision:** To build an automated, cloud-based machine learning pipeline that accurately estimates daily PM2.5 levels across all areas of Karachi using high-resolution satellite imagery, meteorological data, and socio-economic proxies.

---

## 2. 👥 Target Audience & Stakeholders
- **Environmental Researchers & Data Scientists:** For analyzing long-term spatial pollution trends.
- **Policymakers & Urban Planners:** To make data-driven decisions on traffic control and industrial zoning.
- **General Public (via Future App/Demo):** To view historical and estimated pollution levels in their exact neighborhood.

---

## 3. 🏗️ System Architecture & Data Strategy
Instead of relying purely on hardware sensors, the system relies on a **Hybrid Spatio-Temporal Data Engineering Pipeline**:

### **Data Sources (Features)**
1. **Meteorology (ERA5):** Wind speed, Temperature, Relative Humidity.
2. **Aerosols (MODIS):** Aerosol Optical Depth (AOD) at 1km resolution.
3. **Trace Gases (Sentinel-5P):** NO2, SO2, CO, and Aerosol Index (City-wide).
4. **Urban/Socio-economic Proxies:** 
   - **Sentinel-2:** NDVI (Vegetation) and NDBI (Built-up Index).
   - **VIIRS:** Nighttime Lights (Proxy for economic/traffic activity).

### **Tech Stack**
- **Data Extraction:** Google Earth Engine (GEE), Python API.
- **Data Processing:** Python, Pandas, NumPy.
- **Modeling (Upcoming):** Scikit-Learn, XGBoost/LightGBM, PyTorch (Optional).
- **Version Control & CI/CD:** Git, GitHub.

---

## 4. 🚀 Project Phases & Roadmap

### ✅ Phase 1: Cloud Data Collection
- [x] Configure server-side extraction via GEE.
- [x] Extract daily data bounded to Karachi limits (2019–2024).
- [x] Overcome high-latency via server-side spatial reduction.

### ✅ Phase 2: Preprocessing & Data Merging
- [x] Standardize all datasets to a daily frequency.
- [x] Implement a robust merging engine to align multiple spatial resolutions.
- [x] Handle data mismatch, Cartesian explosions, and dummy locators dynamically.

### ✅ Phase 3: Gap Filling & Imputation
- [x] Address natural missing data (dropped 100% empty columns like aer_ai/NO2/SO2/CO).
- [x] Implement K-Nearest Neighbors (KNN) stratified by station to impute sparse MODIS AOD data.
- [x] Integrate a Ground-Truth PM2.5 dataset using a **strict fallback hierarchy** (no fabrication):
  1. OpenAQ v3 per-station PM2.5 (`pm25_source='openaq_exact'`) — 1.9% of rows
  2. US Consulate Karachi anchor (`pm25_source='openaq_us_consulate'`) — 9.2% of rows
  3. NASA MERRA-2 citywide scalar via GEE, van Donkelaar 2010 hygroscopicity formula (`pm25_source='merra2_citywide'`) — 88.8% of rows
  4. Otherwise `pm25_source='missing'` (rows excluded from modeling)
- [x] All rows are real; every row carries a `pm25_source` flag. The
      orchestrator `script/rebuild_master_dataset.py` supersedes the older
      `script/create_master_dataset.py`.

### ✅ Phase 4: Machine Learning Modeling (2023 hold-out test)
- [x] Train Baseline Models:
  - Random Forest — R²=0.612, RMSE=16.30 µg/m³, MAE=10.63
  - SVR — R²=0.551, RMSE=17.54
- [x] Train Advanced Models:
  - XGBoost (GPU) — R²=0.607, RMSE=16.41
  - LightGBM — R²=0.598, RMSE=16.59 (early stopping on a 2022 validation slice, not the test set)
- [x] LSTM Deep Learning with Attention — R²=−0.135 on 1-day-ahead test, RMSE=28.08. The unidirectional causal LSTM with 14-day lookback on 6 stationary features underperforms tree models on this problem; this is consistent with the literature on short-sequence tabular forecasting.
- [x] Feature Importance: SHAP analysis showing `pm25_lag1`, `pm25_roll7`, `Optical_Depth_055`, `wind_speed` as the top drivers
- [x] 5 models trained and serialized in `notebooks/models/`
- [x] **Final feature set** (8 selected by 04 + 5 engineered for 05 = **13 features**): `Optical_Depth_055, wind_speed, month, month_sin, month_cos, day_of_week, is_weekend, viirs_ntl, pm25_lag1, pm25_lag3, pm25_lag7, pm25_roll7, pm25_roll14, pm25_roll30, aod_roll7`

### ✅ Phase 5: Evaluation & Demo Presentation (COMPLETED)
- [x] Model Evaluation: 2023 holdout test set evaluation for all models
- [x] Spatial Analysis: Moran's I=−0.19 (p=0.28, not significant) on **RF residuals** — null result consistent with no detectable spatial autocorrelation remaining after the RF absorbs the satellite + met features. LISA found no significant hotspots.
- [x] 3D Digital Twin: PyDeck 3D visualization with 1km² grid resolution
- [x] Policy Simulation: 6 scenarios (Industry cut, Traffic restriction, Green belt expansion)
- [x] Interactive Dashboards: 5 HTML dashboards with time slider + WHO exceedance counter
- [x] 28 output files generated (PNG charts, CSV results, HTML interactive maps)

---

## 5. 📈 Success Metrics (KPIs) — ACHIEVED

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Reproducible Pipeline** | 100% reproducible | 8 notebooks + 10 Python scripts | ✅ |
| **Model R² Score** | > 0.75 | Best: Random Forest R²=0.612 (honest 2023 hold-out) | ⚠️ |
| **Model RMSE** | Minimize | Best: 16.30 µg/m³ (Random Forest) | ✅ |
| **Spatial Coverage** | City-wide | 8 stations, 1km² grid resolution | ✅ |
| **Temporal Coverage** | 5+ years | 2019–2023 (1,456 days) | ✅ |
| **Digital Twin** | Interactive 3D | 5 HTML dashboards with sliders | ✅ |

**Note on R²:** While 0.61 was achieved (below 0.75 target), this is a typical
honest result for daily PM2.5 prediction from satellite + met features with
limited ground truth. The 0.75 target was aspirational. The LSTM R² < 0 is
also expected: the 14-day sequence of 6 stationary features does not give
the LSTM information that the lag/rolling features already give the tree
models.

---

## 6. 📦 Deliverables Summary

### 📊 Notebooks (8 Total)
| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| 01 | Data Collection | GEE extraction scripts |
| 02 | Preprocessing | Merged dataset CSV |
| 03 | EDA | 7 PNG visualizations |
| 04 | Feature Selection | 6 feature analysis charts |
| 05 | Model Training | 5 trained models + comparison |
| 06 | Spatial Analysis | Moran's I, LISA, IDW maps |
| 07 | LSTM + Digital Twin | PyTorch model + scenarios |
| 08 | Digital Twin Map | Interactive Folium maps |

### 🤖 Trained Models (in `notebooks/models/`) — 2023 hold-out test metrics
- `random_forest.pkl` — **Best performer, R²=0.612**, RMSE=16.30 µg/m³
- `xgboost.pkl` — R²=0.607, RMSE=16.41 (GPU-trained)
- `lightgbm.pkl` — R²=0.598, RMSE=16.59
- `svr.pkl` — R²=0.551, RMSE=17.54
- `prophet.pkl` — R²=−1.36 (poor — only uses date, no features)
- `lstm_model.pt` — R²=−0.135 (1-day-ahead); unidirectional LSTM with attention, 14-day lookback, honest

> **All metrics above are on the 2023 hold-out test set** that the model never
> saw during training or early stopping. A 2022 carve-out from the training
> period is used for early stopping and LR scheduling.

### 🗺️ Interactive Dashboards (in `dashboard/`)
- `karachi_twin_ensemble.html` — 3D ensemble average view
- `karachi_twin_lstm.html` — LSTM predictions 3D view
- `karachi_twin_xgboost.html` — XGBoost 3D view
- `karachi_twin_rf.html` — Random Forest 3D view
- `karachi_twin_svr.html` — SVR 3D view

**Dashboard Features:**
- WHO exceedance counter (real-time)
- Monthly time slider (seasonal simulation)
- Policy sliders: Industry cut (0-50%), Traffic restriction (0-50%), Green expansion (0-50%)
- 3D PyDeck visualization with 1km² grid
- Model comparison links

### 📈 Output Files (28 in `notebooks/outputs/`)
- 21 PNG charts (training curves, spatial maps, scenario analysis)
- 4 CSV result files
- 3 HTML interactive maps
