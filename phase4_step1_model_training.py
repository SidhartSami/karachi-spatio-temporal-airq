"""
Phase 4 — Step 1: Model Training (Random Forest & XGBoost)
==========================================================
What this script does:
  1. Loads the final merged dataset (`modeling_dataset.csv`).
  2. Performs a time-based Train/Test split (e.g., Train: 2019-2022, Test: 2023)
     to simulate real-world forecasting.
  3. Trains a baseline Random Forest Regressor.
  4. Trains an advanced XGBoost Regressor (attempts to use CUDA if available).
  5. Evaluates models using RMSE, MAE, and R^2.
  6. Generates and saves a Feature Importance plot.

Usage:
  python phase4_step1_model_training.py

Dependencies:
  pip install pandas numpy scikit-learn xgboost matplotlib seaborn
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_FILE = "data/processed/modeling_dataset.csv"
FIGURES_DIR = "reports/figures"
MODELS_DIR = "models"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def evaluate_model(name, y_true, y_pred, fit_time):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"--- {name} Results ---")
    logger.info(f"  Training Time : {fit_time:.2f} seconds")
    logger.info(f"  RMSE          : {rmse:.2f} µg/m³")
    logger.info(f"  MAE           : {mae:.2f} µg/m³")
    logger.info(f"  R^2 Score     : {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

def main():
    logger.info(f"Loading dataset: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    
    # ── Feature Engineering & Selection ──────────────────────────────────────
    # Drop identifier columns and the target
    drop_cols = ["date", "station", "pm25", "pm25_source"]
    features = [c for c in df.columns if c not in drop_cols]
    
    logger.info(f"Target: pm25")
    logger.info(f"Features ({len(features)}): {', '.join(features)}")
    
    # ── Time-based Train/Test Split ──────────────────────────────────────────
    # A temporal split is vital for environmental data to prevent data leakage.
    # We train on past data and evaluate on future data.
    train_df = df[df["date"].dt.year <= 2022]
    test_df = df[df["date"].dt.year >= 2023]
    
    X_train, y_train = train_df[features], train_df["pm25"]
    X_test, y_test = test_df[features], test_df["pm25"]
    
    logger.info(f"Train set (2019-2022): {X_train.shape[0]} rows")
    logger.info(f"Test set  (2023)     : {X_test.shape[0]} rows")
    
    # ── Model 1: Random Forest (Baseline) ────────────────────────────────────
    logger.info("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    rf_time = time.time() - start_time
    
    rf_preds = rf_model.predict(X_test)
    evaluate_model("Random Forest", y_test, rf_preds, rf_time)
    
    # ── Model 2: XGBoost (Advanced) ──────────────────────────────────────────
    logger.info("Training XGBoost Regressor...")
    
    # Try CUDA first
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            tree_method="hist",
            device="cuda",
            random_state=42
        )
        logger.info("Attempting XGBoost with CUDA (GPU)...")
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        logger.info("✅ CUDA training successful.")
        
    except Exception as e:
        logger.warning(f"CUDA failed ({str(e)}), falling back to CPU.")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            tree_method="hist",
            device="cpu",
            n_jobs=-1,
            random_state=42
        )
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        logger.info("✅ CPU training successful.")
        
    xgb_preds = xgb_model.predict(X_test)
    evaluate_model("XGBoost", y_test, xgb_preds, xgb_time)
    
    # ── Feature Importance Plot (XGBoost) ────────────────────────────────────
    logger.info("Generating feature importance plot...")
    
    # Get importance dictionary
    importance_dict = xgb_model.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('XGBoost Feature Importance (Weight)')
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'xgboost_feature_importance.png')
    plt.savefig(fig_path, dpi=300)
    logger.info(f"✅ Feature importance saved → {fig_path}")

if __name__ == "__main__":
    main()
