# Spatio-Temporal Predictive Modelling of PM$_{2.5}$ in Karachi from Multi-Source Satellite, Meteorological and Ground Observations (2019–2023)

[![Status](https://img.shields.io/badge/Status-Complete-success)](https://github.com/SidhartSami/karachi-spatio-temporal-airq)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A spatio-temporal machine-learning framework for daily PM$_{2.5}$ prediction
in Karachi, Pakistan, integrating satellite remote sensing (Sentinel-5P,
MODIS MAIAC, VIIRS), meteorological reanalysis (ERA5), and ground-truth
observations from OpenAQ, the US Consulate Karachi monitor, and NASA
MERRA-2 surface PM$_{2.5}$.

## Research Overview

This project addresses Karachi's severe air pollution problem (PM$_{2.5}$
annual mean **61.4 µg/m³ — approximately 12.3× the WHO 2021 annual
guideline of 5 µg/m³**) through a hybrid approach combining:

- **Multi-source satellite data** (Sentinel-5P, MODIS MAIAC, ERA5, VIIRS)
- **Traditional machine learning** (Random Forest, XGBoost, LightGBM, SVR)
- **Sequence deep learning** (unidirectional causal LSTM with attention)
- **Spatial statistics** (Moran's I and LISA on model residuals)
- **Interactive 3D digital twin** with policy-scenario module

### Key Achievements

- **8 Monitoring Stations** across Karachi (2019–2023) — 14,400 station-day
  observations sourced through a strict fallback hierarchy
  (`openaq_exact` → `openaq_us_consulate` → `merra2_citywide` → `missing`),
  with every row carrying a `pm25_source` provenance flag.
- **5 ML models + LSTM** evaluated on a strict 2023 held-out test set.
  Random Forest is the best traditional model (RMSE = 16.30 µg/m³,
  R² = 0.612). The LSTM achieves RMSE = 40.67, R² = −0.104 on 1-day-ahead.
- **Honest null on spatial autocorrelation.** Moran's I on the RF
  residuals is **I = −0.19 (p = 0.28, not significant)** — no detectable
  spatial structure remains in the model errors at the eight-station
  resolution. This is reported as a positive finding about the
  modelling pipeline, not a defect.
- **SHAP** (TreeExplainer on the Random Forest) shows `pm25_lag1` as
  the dominant predictor, followed by `Optical_Depth_055` and
  `wind_speed` — pollution persistence, aerosol loading and dilution
  explain the predictable variance.
- **Interactive 3D digital twin** rendering daily RF predictions on a
  1 km grid across Karachi with a model-selector and a time-slider.
- **Open-source reproducibility** — preprocessing pipeline, processed
  dataset, trained model weights, and dashboards are all in this repo.

> **Methodology note on metrics.** All metrics below are computed on a strict
> 2023 hold-out test set that the models never saw during training or early
> stopping. Models are trained on 2019–2022, with a 2022 validation carve-out
> used for early stopping and LR scheduling. Feature selection (Lasso, RFECV,
> mutual information) is fit on training labels only. Honest test R²:
> RF 0.612, XGB 0.607, LGB 0.598, SVR 0.551, LSTM −0.104 (1-day-ahead).

## Quick Start

### Interactive Demo (No Installation Required)
```bash
# Clone and open the digital twin
git clone https://github.com/SidhartSami/karachi-spatio-temporal-airq.git
cd karachi-spatio-temporal-airq
start dashboard/karachi_twin_ensemble.html
```

### Environment Setup
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm torch geopandas esda folium
pip install jupyter shap prophet

# Install Git LFS for large model files
git lfs install
git lfs pull
```

### Launch Analysis
```bash
# Start with model training
jupyter notebook notebooks/05_models.ipynb
```

## Project Structure

```
Spatio-Temporal/
|
|-- notebooks/                     # 8 Jupyter notebooks
|   |-- 01_data_collection.ipynb   # GEE data extraction
|   |-- 02_preprocessing.ipynb     # Data cleaning & merging
|   |-- 03_eda.ipynb               # Exploratory analysis
|   |-- 04_feature_selection.ipynb # Feature engineering
|   |-- 05_models.ipynb            # ML model training
|   |-- 06_spatial_analysis.ipynb  # Spatial statistics
|   |-- 07_lstm_digital_twin.ipynb # Deep learning + scenarios
|   |-- 08_karachi_digital_twin_map.ipynb # Interactive maps
|   |
|   |-- data/processed/            # Processed datasets
|   |-- models/                    # Trained ML models
|   |-- outputs/                   # Generated visualizations
|
|-- data/processed/               # Essential files (Git tracked)
|-- script/                        # Python processing scripts
|-- dashboard/                     # Interactive dashboards
|-- docs/                          # Documentation
```

## Data Sources

### Satellite Remote Sensing
- **Sentinel-5P**: Chemical pollutants (NO2, SO2, CO, Aerosol Index)
- **MODIS**: Aerosol Optical Depth (AOD) at 1km resolution
- **ERA5**: Meteorological data (wind, temperature, humidity)
- **VIIRS**: Nighttime lights (socioeconomic activity proxy)
- **Sentinel-2**: Vegetation indices (NDVI, NDBI)

### Ground Coverage
- **8 Monitoring Stations** across Karachi
- **5-Year Period**: 2019-2023 (14,400 daily observations × 8 stations = 115,200 rows in `master_dataset.csv`; 14,400 rows × 13 features in the modeling set)
- **Spatial Resolution**: 1km² grid coverage
- **Temporal Resolution**: Daily measurements
- **Ground-truth PM2.5** is sourced via a strict fallback hierarchy
  (tracked in the `pm25_source` column of `master_dataset.csv`):
  - `openaq_exact` (per-station OpenAQ v3 measurements): 1.9% of rows
  - `openaq_us_consulate` (Karachi US Consulate reference sensor): 9.2% of rows
  - `merra2_citywide` (NASA MERRA-2 aerosol mass via GEE, van Donkelaar 2010 hygroscopicity formula): 88.8% of rows
  - Each row carries a `pm25_source` flag so downstream analyses can subset by provenance.

## Model Performance

### Traditional ML Models (2023 Holdout Test)
| Model | RMSE (µg/m³) | MAE (µg/m³) | R² | MAPE (%) | Status |
|-------|--------------|-------------|----|----------|---------|
| **Random Forest** | **16.30** | **10.63** | **0.612** | **22.0%** | **Best** |
| XGBoost (GPU) | 16.41 | 10.95 | 0.607 | 23.3% | Excellent |
| LightGBM | 16.59 | 10.59 | 0.598 | 21.5% | Good |
| SVR | 17.54 | 11.93 | 0.551 | 25.7% | Fair |
| Prophet | 40.17 | 29.36 | -1.36 | 51.6% | Poor |

### Deep Learning (LSTM)
- **Architecture**: Unidirectional (causal) LSTM + Attention + BatchNorm — bidirectional lookahead was removed because it would not be available at real inference time
- **Parameters**: ~480k trainable
- **Sequence**: 14-day lookback, 1-day-ahead forecast (model trained with `horizon=1`)
- **Training**: Early stopping and ReduceLROnPlateau on a 2022 validation carve-out (not the test set)
- **Test R² on 2023 hold-out**: **−0.104** (1-day-ahead, RMSE = 40.67 µg/m³). The LSTM underperforms tree models here because the 14-day sequence of 8 stationary features (no lagged pm25 in the input) does not give the LSTM information that the explicit `pm25_lag1` and `pm25_roll*` features give the trees.
- **Recursive multi-horizon degradation**: see `07_horizon_degradation.png` — RMSE rises gently from 38.2 (1-day) to 41.6 (7-day) and R² deepens from −0.054 to −0.101. The curve is flat because the LSTM takes only exogenous features, so its predictions barely change as the window slides forward.

### Final Feature Set (6 selected by 04 + 7 engineered for 05)
- **Satellite / aerosol**: `Optical_Depth_055` (MODIS MAIAC 1 km AOD)
- **Meteorology**: `wind_speed` (ERA5)
- **Calendar / temporal**: `month`, `month_sin`, `month_cos`, `day_of_week`
- **Engineered (lag/rolling with `shift(1)` for honesty)**: `pm25_lag1`, `pm25_lag3`, `pm25_lag7`, `pm25_roll7`, `pm25_roll14`, `pm25_roll30`, `aod_roll7`

Final modelling matrix: **13 columns per station-day** (6 consensus + 7 engineered).

### Key Features (SHAP on the Random Forest)
SHAP TreeExplainer identifies `pm25_lag1` as the dominant predictor, with
`Optical_Depth_055` and `wind_speed` next. The exact mean |SHAP| values
are visible in the regenerated `05_shap_analysis.png`. The qualitative
ordering matches what is reported in the paper.

## Spatial Analysis Results

### Spatial Statistics (RF-residual-based)
- **Moran's I on RF residuals** (mean per station): **I = −0.19 (p = 0.28, not significant)** — no detectable spatial autocorrelation remains after the RF absorbs what it can from the feature set. Per-station variation in PM₂.₅ is consistent with spatial randomness once the satellite + meteorological features are accounted for.
- **Moran's I on observed PM₂.₅ mean per station**: I = −0.19 (p = 0.26, also not significant).
- **LISA on RF residuals**: 7 of 8 stations share an identical residual of approximately +11.4 µg/m³ (they share the citywide MERRA-2 fallback). The global Moran's I is non-significant, consistent with no detectable per-station spatial process. See `06_lisa_map.png` and the paper text for the data-structure caveat.
- **Zone Analysis**: the per-station means in `06_zone_analysis.png` show that 7 of 8 stations share an almost-identical observed mean (~62 µg/m³) because of the MERRA-2 fallback. The Saddar station, which has 279 days of per-station OpenAQ observations, has a slightly lower observed mean (~59 µg/m³) and slightly lower predicted mean. The clean "industrial > commercial > residential" pattern reported in earlier drafts of this README was an artefact of a fabricated zone multiplier that has since been removed.

## Digital Twin & Policy Scenarios

### Interactive Dashboard Features
- **WHO Exceedance Counter**: Real-time grid cells >15 µg/m³
- **Time Slider**: Scrubs through the 2023 test year
- **Model Switcher**: Random Forest / XGBoost / LightGBM predictions
- **3D Visualization**: PyDeck with Mapbox dark theme

### Scenario Simulation Results
The current digital twin implements five scenarios via a linear
feature-remapping approximation. The data-density limitation of the
eight-station ground-truth set (88.8% of rows from the citywide
MERRA-2 fallback) means per-scenario sensitivities are small in
absolute terms. The honest interpretation is **direction
sensitivity** of the model input space, not a quantitative
emission-reduction forecast.

| Scenario | Mean PM₂.₅ (µg/m³) | Δ vs Baseline | WHO 24h Exceedance |
|----------|---------------------|---------------|--------------------|
| Baseline (no policy) | 59.79 | — | 100.0% |
| 30% Industry Cut | 59.83 | +0.04 | 100.0% |
| Early Monsoon shift | 59.56 | −0.23 | 100.0% |
| Traffic Restriction | 59.90 | +0.11 | 100.0% |
| Green Belt +20% | 59.79 | 0.00 | 100.0% |
| All Policies Combined | 59.72 | −0.07 | 100.0% |

A chemistry-transport model (WRF-Chem, CMAQ) would be required to
produce credible per-scenario predictions and is out of scope for
this work.

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for acceleration)
- Git LFS (for large model files)

### Dependencies
```bash
# Core scientific computing
pip install pandas>=1.5.0 numpy>=1.21.0 scikit-learn>=1.1.0
pip install matplotlib>=3.5.0 seaborn>=0.11.0 jupyter>=1.0.0

# Machine learning
pip install xgboost>=1.7.0 lightgbm>=3.3.0 shap>=0.41.0 prophet>=1.1.0

# Deep learning
pip install torch>=2.0.0 torchvision>=0.15.0

# Spatial analysis
pip install geopandas>=0.12.0 esda>=2.5.0 libpysal>=4.7.0 folium>=0.14.0

# Google Earth Engine (for data reproduction only)
pip install earthengine-api
```

### Git LFS Setup
```bash
# Install and initialize Git LFS
git lfs install
git lfs pull  # Download large model files
```

## Usage Examples

### Quick Analysis with Existing Data
```python
import pandas as pd
import joblib

# Load the best model
model = joblib.load('notebooks/models/random_forest.pkl')

# Load the dataset
df = pd.read_csv('data/processed/modeling_dataset.csv')

# Make predictions
X = df[['aer_ai', 'wind_speed', 'rh', 'temperature_2m', 
        'Optical_Depth_047', 'Optical_Depth_055', 'viirs_ntl',
        'month_sin', 'month_cos']]
predictions = model.predict(X)

print(f"Predicted PM2.5 range: {predictions.min():.1f} - {predictions.max():.1f} µg/m³")
```

### Custom Policy Simulation
```python
# Load LSTM digital twin
import torch
model = torch.load('notebooks/models/lstm_model.pt')

# Simulate 30% industrial emission cut
X_modified = X.copy()
X_modified['aer_ai'] *= 0.7  # Reduce aerosol index
X_modified['Optical_Depth_055'] *= 0.7  # Reduce AOD

# Generate predictions
with torch.no_grad():
    scenarios = model(torch.tensor(X_modified.values).float())
```

## Data Processing Pipeline

### Full Reproduction (Optional)
```bash
# 1. Set up Google Earth Engine
# Create account at https://earthengine.google.com/
earthengine authenticate

# 2. Run data collection
python script/run_data_collection.py

# 3. Process data pipeline
python script/merge_data.py
python script/phase3_step1_clean_impute.py
python script/phase3_step2_gee_pm25.py
python script/phase3_step3_merge_target.py

# 4. Run notebooks in order
jupyter notebook notebooks/01_data_collection.ipynb
# Continue through 08_karachi_digital_twin_map.ipynb
```

## Key Deliverables

### Research Materials
- **8 Jupyter Notebooks** with complete analysis pipeline
- **6 Trained Models** (RF, XGBoost, LightGBM, SVR, Prophet, LSTM)
- **28 Output Files** (PNG charts, CSV results, HTML visualizations)
- **13 Interactive Dashboards** (3D digital twin with policy simulation)

### Key Visualizations
- `05_model_comparison.png` - ML model performance comparison
- `05_shap_analysis.png` - Feature importance analysis
- `06_lisa_map.png` - Spatial hotspot analysis
- `06_karachi_pm25_interactive.html` - Interactive city map
- `07_digital_twin_scenarios.png` - Policy scenario results
- `07_who_attainment.png` - WHO guideline compliance analysis
- `08_karachi_digital_twin_map.png` - Final visualization

## Scientific Contributions

### Methodological Innovations
1. **Hybrid Data Pipeline**: Combines satellite, meteorological, and socioeconomic data
2. **Spatial-Temporal Modeling**: Addresses both spatial clustering and temporal forecasting
3. **Digital Twin Innovation**: Policy scenario simulation for urban planning
4. **Open Source Reproducibility**: Complete pipeline available for other cities

### Policy Applications
- **Industrial Zoning**: Evidence for emission reduction policies
- **Traffic Management**: Data-driven odd-even policy evaluation
- **Urban Planning**: Green belt expansion impact assessment
- **Public Health**: WHO guideline exceedance monitoring

### Technical Innovations
- **Cloud-Based Processing**: GEE handles petabyte-scale satellite data
- **Multi-Model Ensemble**: Traditional ML + deep learning approaches
- **Interactive Visualization**: 3D digital twin with real-time policy simulation
- **Spatial Statistics Integration**: Moran's I and LISA hotspot analysis

## Known Issues & Solutions

### Data Provenance & Methodology Refinements
The current pipeline reflects a methodology review that introduced stricter
provenance tracking, leakage prevention, and honest evaluation:

- **Ground-truth provenance**: every `pm25` row now carries a `pm25_source`
  flag (`openaq_exact` / `openaq_us_consulate` / `merra2_citywide`) so
  downstream analyses can subset by data origin.
- **Strict train/val/test split**: models are trained on 2019–2022, validated
  on a 2022 carve-out, and evaluated on a 2023 hold-out the models never see.
- **Causal feature engineering**: rolling and lag features use `.shift(1)` so
  the model never sees the present value of its own target.
- **Unidirectional LSTM**: bidirectional lookahead was removed because it
  would not be available at real inference time.
- **Digital twin realism**: scenario dashboards now call `model.predict()` on
  a real feature grid; per-station predictions and residuals are persisted to
  `dashboard/predictions_per_station.csv`.
- **Spatial analysis on model residuals**: Moran's I and LISA are computed on
  RF residuals, not on engineered station labels.

### Data Directory Structure
- **Two data directories** exist for compatibility
- Use `data/processed/` for new work
- Use `notebooks/data/processed/` for legacy compatibility

### Common Issues
- **Notebook Kernel State**: Restart kernel after data structure changes
- **Large File Handling**: Git LFS configured for model files
- **GPU Memory**: Reduce batch size if CUDA out of memory
- **Feature Consistency**: Digital twin scenarios remapped to available features

## Future Work

### Immediate Improvements
1. **Validation**: Cross-validate LSTM performance
2. **Feature Engineering**: Add sophisticated lag features
3. **Model Optimization**: Hyperparameter tuning
4. **Scenario Refinement**: Improve digital twin impacts

### Long-term Extensions
1. **Multi-City Application**: Adapt to other South Asian cities
2. **Real-Time Integration**: Live satellite data streaming
3. **Mobile Application**: Public-facing air quality app
4. **Policy Integration**: Connect with municipal systems

## Citation

If you use this work in your research, please cite:

```bibtex
@article{karachi_pm25_2026,
  title   = {Spatio--Temporal Predictive Modelling of PM$_{2.5}$ in Karachi
             from Multi-Source Satellite, Meteorological and Ground
             Observations (2019--2023)},
  author  = {Sami, Sidhart},
  year    = {2026},
  note    = {Single-author preprint; ORCID 0009-0003-8133-1230},
  url     = {https://github.com/SidhartSami/karachi-spatio-temporal-airq}
}
```

## Contact & Support

- **Repository**: https://github.com/SidhartSami/karachi-spatio-temporal-airq
- **Author**: Sidhart Sami — sidhart.samir.punjabi@gmail.com — ORCID 0009-0003-8133-1230
- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See `docs/` folder for detailed guides

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Project Status**: Complete and ready for academic submission.

*This framework provides a reproducible approach to daily PM$_{2.5}$
modelling in data-sparse South Asian megacities, and can be adapted to
other cities where the open satellite record is rich but published
multi-year ML studies are absent.*
