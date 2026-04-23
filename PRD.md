# 📄 Product Requirements Document (PRD)
**Project Name:** Karachi Spatio-Temporal Air Quality Estimator  
**Status:** In Development (Phase 3 of 5)  
**Last Updated:** 2026-04-23  

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

### ⏳ Phase 3: Gap Filling & Imputation (Next Step)
- [ ] Address natural missing data (e.g., satellite cloud-masking during monsoons).
- [ ] Implement interpolation, K-Nearest Neighbors (KNN), or MICE to impute sparse S5P/MODIS data.
- [ ] Integrate a Ground-Truth PM2.5 dataset as the target variable for the model.

### 📅 Phase 4: Machine Learning Modeling
- [ ] Train Baseline Models (Linear Regression, Random Forest).
- [ ] Train Advanced Models (XGBoost, Spatio-Temporal Neural Networks).
- [ ] Perform Feature Importance analysis (Which factor drives Karachi's pollution the most?).

### 📅 Phase 5: Evaluation & Demo Presentation
- [ ] Evaluate model on unseen data (Calculate $R^2$, RMSE, MAE).
- [ ] Generate spatial heatmaps of pollution.
- [ ] Finalize repository documentation and the presentation sequence.

---

## 5. 📈 Success Metrics (KPIs)
- **Technical Pipeline:** 100% reproducible pipeline from data download to model prediction.
- **Model Accuracy:** Achieve an $R^2$ score $> 0.75$ and minimize Root Mean Square Error (RMSE) against the ground truth.
- **Robustness:** Model must successfully handle missing satellite days using the gap-filling logic.
