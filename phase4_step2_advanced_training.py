"""
Phase 4 — Step 2: Advanced Modeling & Hyperparameter Tuning
===========================================================
What this script does:
  1. Loads dataset and applies Standardization (StandardScaler).
  2. Implements 4 distinct ML Models (Linear, SVR, RF, XGB).
  3. Performs Hyperparameter Tuning on XGBoost using GridSearchCV.
  4. Evaluates all models (RMSE, MAE, R^2) and saves a comparative plot.

Usage:
  python phase4_step2_advanced_training.py
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

INPUT_FILE = "data/processed/modeling_dataset.csv"
FIGURES_DIR = "reports/figures"

def evaluate_model(name, y_true, y_pred, fit_time):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logger.info(f"[{name}] Time: {fit_time:.2f}s | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "Time_s": fit_time}

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    
    # ── 1. Feature Engineering & Scaling ─────────────────────────────────────
    drop_cols = ["date", "station", "pm25", "pm25_source"]
    features = [c for c in df.columns if c not in drop_cols]
    
    train_df = df[df["date"].dt.year <= 2022]
    test_df = df[df["date"].dt.year >= 2023]
    
    X_train_raw, y_train = train_df[features], train_df["pm25"]
    X_test_raw, y_test = test_df[features], test_df["pm25"]
    
    # Normalization (Required for Linear Regression and SVR)
    logger.info("Applying StandardScaler (Normalization)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    results = []
    
    # ── 2. Model 1: Linear Regression ────────────────────────────────────────
    logger.info("Training Model 1: Linear Regression...")
    lr = LinearRegression()
    t0 = time.time()
    lr.fit(X_train, y_train)
    t_fit = time.time() - t0
    results.append(evaluate_model("Linear Regression", y_test, lr.predict(X_test), t_fit))

    # ── 3. Model 2: Support Vector Regressor (SVR) ───────────────────────────
    # Subsampling SVR training since it scales poorly (O(n^2))
    logger.info("Training Model 2: Support Vector Regressor (SVR)...")
    svr = SVR(kernel='rbf', C=10, gamma='scale')
    t0 = time.time()
    svr.fit(X_train[:5000], y_train[:5000]) # Train on subset for speed
    t_fit = time.time() - t0
    results.append(evaluate_model("SVR (Subset)", y_test, svr.predict(X_test), t_fit))

    # ── 4. Model 3: Random Forest ────────────────────────────────────────────
    logger.info("Training Model 3: Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train, y_train)
    t_fit = time.time() - t0
    results.append(evaluate_model("Random Forest", y_test, rf.predict(X_test), t_fit))

    # ── 5. Model 4: XGBoost with Hyperparameter Tuning ───────────────────────
    logger.info("Training Model 4: XGBoost (with GridSearchCV)...")
    # Base model with GPU if available
    xgb_base = xgb.XGBRegressor(tree_method="hist", device="cuda", random_state=42)
    
    # Define a small grid to save time during demo
    param_grid = {
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=1
    )
    
    t0 = time.time()
    try:
        grid_search.fit(X_train, y_train)
        using_gpu = True
    except Exception as e:
        logger.warning(f"CUDA failed for GridSearch, falling back to CPU: {e}")
        xgb_base = xgb.XGBRegressor(tree_method="hist", device="cpu", n_jobs=-1, random_state=42)
        grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3)
        grid_search.fit(X_train, y_train)
        using_gpu = False
        
    t_fit = time.time() - t0
    best_xgb = grid_search.best_estimator_
    logger.info(f"Best XGBoost Params: {grid_search.best_params_}")
    
    results.append(evaluate_model(f"XGBoost (Tuned)", y_test, best_xgb.predict(X_test), t_fit))
    
    # ── 6. Comparative Analysis Plot ─────────────────────────────────────────
    res_df = pd.DataFrame(results)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for RMSE
    sns.barplot(x='Model', y='RMSE', data=res_df, ax=ax1, color='lightcoral', alpha=0.8, label='RMSE (Lower is better)')
    ax1.set_ylabel('RMSE (µg/m³)')
    ax1.set_ylim(0, res_df['RMSE'].max() * 1.2)
    
    # Line plot for R2 on secondary axis
    ax2 = ax1.twinx()
    sns.lineplot(x='Model', y='R2', data=res_df, ax=ax2, color='darkblue', marker='o', linewidth=2, markersize=8, label='R² (Higher is better)')
    ax2.set_ylabel('R² Score')
    ax2.set_ylim(0, 1.0)
    
    plt.title('Machine Learning Models Comparison', fontsize=16)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=300)
    logger.info("✅ Model comparison plot saved to reports/figures/model_comparison.png")

if __name__ == "__main__":
    main()
