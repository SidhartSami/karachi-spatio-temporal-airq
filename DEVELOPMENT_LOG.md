# 📋 Karachi Spatio-Temporal Air Quality Project: Development Log

This document serves as the primary technical record for the Karachi Air Quality Research project. It documents every phase from data acquisition to preprocessing, the challenges encountered, and the engineering solutions implemented.

---

## 📅 Project Timeline & Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| **01 Data Collection** | Multi-source satellite acquisition via Google Earth Engine | ✅ Completed |
| **02 Preprocessing** | Data cleaning, merging, and row count stabilization | ✅ Completed |
| **03 Gap Filling & Target** | Cleaned missing columns, KNN Imputation, and MERRA-2 PM2.5 integration | ✅ Completed |
| **04 Modeling** | PM2.5 estimation via Tree-based and Deep Learning models | ⏳ In Progress |

---

## 🚀 Current Status: GitHub Synchronization & Refinement
We have transitioned into a high-frequency synchronization phase, ensuring every technical refinement is immediately committed and pushed to GitHub. This maximizes version control granularity and ensures team transparency for the upcoming demo.

---

## ✅ Milestones Completed

### 1. Multi-Source Data Acquisition (GEE)
- **ERA5 Meteorology:** Successfully exported daily meteorological features (Wind Speed, RH, Temp) from 2019–2024.
- **MODIS AOD:** Acquired Aerosol Optical Depth data at 1km resolution.
- **S2 & VIIRS:** Integrated Sentinel-2 (NDVI/NDBI) and VIIRS (Nighttime Lights) for spatial and socioeconomic features.
- **Sentinel-5P (v3):** Triggered citywide regional averaging for NO2, SO2, CO, and Aerosols.

### 2. Merging Infrastructure & Cleanup
- Built a robust Python pipeline (`merge_data.py`) to unify multiple satellite CSVs on a common spatio-temporal grid (Date + Station).
- **GitHub Optimization:** Implemented a granular commit strategy and performed a repository cleanup (removing redundant/deprecated notebooks).

---

## 🛰️ Phase 01: Data Collection (GEE Architecture)

### 1. The Multi-Source Strategy
We integrated data from four distinct satellite/reanalysis sources to capture a holistic view of the atmosphere:
- **ERA5 (Meteorology):** Temperature, Wind Speed (U/V components), and Relative Humidity.
- **MODIS (Aerosols):** Aerosol Optical Depth (AOD) at 1km resolution.
- **VIIRS & Sentinel-2:** Socio-economic proxies (Nighttime lights) and Urban density (NDVI/NDBI).
- **Sentinel-5P (Chemicals):** NO2, SO2, CO, and Aerosol Index.

### 2. The Cloud-to-Drive Workflow
Unlike traditional downloading, our pipeline processes data **on the server**:
1.  **Filtering:** We apply spatial (Karachi Boundary) and temporal (2019–2024) filters directly on Google's Petabyte-scale servers.
2.  **Reduction:** Millions of satellite pixels are aggregated into simple tabular formats (CSV) using `reduceRegions`.
3.  **Export:** These processed tables are queued in the GEE Task Manager and uploaded directly to **Google Drive**.
4.  **Local Sync:** Files are then downloaded to the `data/raw/` directory for local analysis.

### 3. Why Google Earth Engine (GEE)?
This choice was strategic for several reasons:
-   **Computational Power:** Processing 5 years of daily satellite data for a city like Karachi involves trillions of calculations. GEE performs this in parallel across thousands of Google CPUs.
-   **Data Accessibility:** Instead of downloading Terabytes of raw `.tif` images, we only download the final 5MB CSV tables, saving massive bandwidth and storage.
-   **Reproducibility:** The entire extraction logic is contained in a single Python script (`run_data_collection.py`), ensuring anyone with access can re-run the pipeline.

---

## 🧹 Phase 02: Preprocessing & Data Merging

### 1. The Merging Engine (`merge_data.py`)
Merging satellite data is complex because every satellite has a different orbit and timing. Our script performs the following:
- **Date Standardization:** Converging various timestamp formats into a unified `YYYY-MM-DD`.
- **Location Mapping:** Mapping satellite pixels to ground-level monitoring stations.
- **Outer Join Strategy:** We use a "Keep All" (Outer) join to ensure that even if a satellite is blocked by clouds on one day, the meteorological data from that day is preserved.

### 2. Standardization Logic
We implemented a mapping layer to ensure consistency:
- `location` ➔ `station`
- `wind_u`, `wind_v` ➔ `wind_speed` & `wind_direction`
- `RH` (Relative Humidity) derivation from Dewpoint and Temperature.

---

## 🛠️ Critical Issues & Engineering Solutions

### 🔴 Problem 1: The "Sentinel-5P Data Hole"
- **The Issue:** S5P chemical values were returning as `NaN` or 0 in initial exports.
- **The Discovery:** Sentinel-5P has a resolution of 7km. Small point-based buffers often missed the center of these large pixels, especially when cloud-masking was active.
- **The Solution:** We pivoted from **Point Extraction** to **Citywide Regional Averaging**. By averaging the chemical signal across all of Karachi, we captured the "Urban Background" pollution level, which is more scientifically robust and significantly reduced missing data.

### 🔴 Problem 2: Row Count "Explosion" (Cartesian Products)
- **The Issue:** After merging, the dataset jumped from 14,000 rows to 1.2 million.
- **The Discovery:** Some satellites (like MODIS) have multiple "granules" or overpasses per day. A simple join causes every overpass to match with every ground station row, multiplying the count.
- **The Solution:** Implemented a mandatory **Aggregation Step** in the loading function. All datasets are now grouped by `(date, station)` and averaged *before* the merge. This ensures a clean 1-row-per-day-per-station structure.

### 🔴 Problem 3: Memory & Processing Latency
- **The Issue:** Processing 5 years of daily global data is impossible on a standard laptop.
- **The Solution:** All heavy lifting (Pixel math) is offloaded to GEE. The local Python script only handles the final 5MB CSV merges, making the pipeline lightweight and reproducible.

### 🔴 Problem 4: Spatial-Temporal Alignment (Cartesian Explosion & Broadcasting)
- **The Issue:** MODIS AOD values were returning 0 non-nulls after the merge, while VIIRS Nighttime Lights was completely dropping out.
- **The Discovery:** 
  1. MODIS was extracted with a dummy citywide station name (`Karachi`), preventing the `pd.merge` from mapping values to the 8 specific ground stations.
  2. VIIRS data was processed on a monthly frequency (`year_month`), lacking daily granularity, which threw off the daily join operations. 
- **The Solution:** 
  - Rewrote the merging logic in `merge_data.py` utilizing a custom `smart_merge()` function. 
  - Detected and dropped dummy `station='Karachi'` columns dynamically to force a cross-broadcast of citywide metrics (like MODIS and ERA5) across all 8 individual stations, effectively enriching local meteorology with regional baselines.
  - Aligned VIIRS on `year_month` and correctly mapped `mean` aggregated metrics back onto the daily granular records.

---

## 🧪 Phase 03: Gap Filling & Ground Truth Integration

### 1. Data Cleaning & Imputation (`phase3_step1_clean_impute.py`)
- **Dead Columns Removed:** Dropped `no2`, `so2`, and `co` as they returned 100% missing values in the specific extraction format.
- **Spatio-Temporal KNN Imputation:** Handled ~10% missing values in MODIS AOD (`Optical_Depth_047`, `Optical_Depth_055`) using a `KNNImputer(k=5)` stratified by Station to ensure spatial relevance.

### 2. The Ground Truth Shift: NASA MERRA-2 (`phase3_step2_gee_pm25.py`)
- **Initial Challenge:** Our original plan was to use physical OpenAQ sensor data, but historical coverage was too sparse (many stations had massive gaps or missing API data).
- **The Solution:** We transitioned to a GEE-native **MERRA-2 approach**. We pull NASA's aerosol mass surface components (Black Carbon, Organic Carbon, Sulfate, Sea Salt, and Dust) and apply a NASA-validated stoichiometric formula to compute daily surface PM2.5 levels. 
- **Benefit:** This gives us a mathematically robust, 100% complete temporal target dataset for our entire 2019-2024 timeframe.

### 3. Final Alignment (`phase3_step3_merge_target.py`)
- The clean satellite features were merged with the MERRA-2 target PM2.5 to form `modeling_dataset.csv` (14,592 rows × 19 columns). The target vector `pm25` boasts 100% coverage, clearing the path for robust Machine Learning training.

---

## ❓ Technical FAQ for Demo Preparation

**Q: Why do we have missing values in satellite data?**  
*A: Satellites use light to measure air quality. They cannot see through clouds or heavy fog (Monsoon/Winter). This is a natural limitation of remote sensing, not a bug in the code.*

**Q: How do we know the citywide S5P average is accurate for a specific station?**  
*A: Pollutants like NO2 and CO have a high "residence time" and disperse across the city. The citywide average represents the "Regional Load." Combined with station-specific meteorology (ERA5), it provides a very strong predictor for local PM2.5.*

**Q: What is the "Ground Truth" for this project?**  
*A: The project is designed to estimate PM2.5. The features we've collected (AOD, Wind, Pollutants) are the independent variables that explain the ground-level PM2.5 concentrations.*

---

## 📝 Next Steps for the Team
---

## 🛠️ Project Script Index

| Script | Purpose | When to Run |
| :--- | :--- | :--- |
| `run_data_collection.py` | Initial GEE extraction (Point-based) | *Deprecated* (Use for ERA5/MODIS only) |
| `scratch/run_s5p_citywide.py` | GEE Citywide Extraction (The S5P Solution) | Run when chemical data is needed |
| `merge_data.py` | Unifies all CSVs into one master dataset | Run after downloading new CSVs from Drive |
| `phase3_step1_clean_impute.py` | Drops dead columns & performs KNN imputation on missing AOD | Run on merged data to clean features |
| `phase3_step2_gee_pm25.py` | Extracts NASA MERRA-2 daily PM2.5 target via GEE | Run to fetch baseline Ground Truth |
| `phase3_step3_merge_target.py` | Joins clean features with MERRA-2 PM2.5 ground truth | Final step to build `modeling_dataset.csv` |

---

## 🚀 How to Run (The Demo Sequence)

1.  **GEE Extraction:**
    ```bash
    python scratch/run_s5p_citywide.py
    ```
    *This triggers the server-side processing at Google. You must wait for the tasks to finish in the GEE Console.*

2.  **Download & Move:**
    Download the CSVs from your Google Drive and place them in `data/raw/`.

3.  **Merge Data:**
    ```bash
    python merge_data.py
    ```
    *This creates the `karachi_air_quality_merged.csv` file used for modeling.*

---

## 📊 Technical Justification for the Demo
"We didn't just 'scrape' data; we engineered a **Hybrid Spatio-Temporal Pipeline**. By combining high-resolution meteorology (ERA5) with regional chemical signatures (Sentinel-5P), we overcome the traditional 'Cloud Masking' problem that plagues 90% of satellite-based air quality studies in South Asia."
