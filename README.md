# Spatio-Temporal Air Quality Analysis: Karachi (2019–2024)

A robust, publication-quality pipeline for analyzing and predicting air quality in Karachi, Pakistan using multi-source satellite data and advanced machine learning techniques.

## 📌 Project Overview
This project integrates high-resolution satellite imagery and meteorological data to construct a comprehensive spatio-temporal dataset for Karachi. It features an automated data-merging pipeline, sophisticated preprocessing workflows, and predictive modeling tailored for urban air quality monitoring.

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
- `notebooks/01_data_collection.ipynb`: Google Earth Engine (GEE) export orchestration.
- `notebooks/02_eda.ipynb`: Temporal cycles, spatial rankings, and correlation analysis.
- `notebooks/03_preprocessing.ipynb`: Smart imputation, scaling, and final dataset generation.
- `notebooks/04_modeling.ipynb`: Comparative analysis using Tree-based models, SVR, and LSTM.
- `data/`: Directory for raw and processed datasets (ignored by git for size).
- `scratch/`: Utility scripts for GEE task monitoring and debugging.

## 🚀 Getting Started
1. **GEE Authentication:** Ensure you have a Google Earth Engine account and a project ID.
2. **Environment Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Execution:** Run the notebooks in sequential order (`01` through `04`).

## 📊 Methodology Highlights
The pipeline is designed to be resilient to the "missing pixel" problem common in satellite air quality monitoring. By buffering station coordinates and using 3-stage imputation, we achieve a near-continuous daily dataset suitable for deep learning (LSTM) and traditional regression models.
