"""
Phase 4 — Step 0: Exploratory Data Analysis (EDA)
=================================================
Generates core visualizations for the Data Science Research Paper.

Outputs saved to: reports/figures/
  1. correlation_heatmap.png
  2. pm25_distribution.png
  3. monthly_pm25_trend.png
  4. aod_vs_pm25_scatter.png
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INPUT_FILE = "data/processed/modeling_dataset.csv"
FIGURES_DIR = "reports/figures"

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    logger.info(f"Loading data from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Correlation Heatmap
    logger.info("Generating Correlation Heatmap...")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, 
                cbar_kws={"shrink": 0.8}, square=True, annot_kws={"size": 8})
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # 2. PM2.5 Distribution
    logger.info("Generating PM2.5 Distribution Plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df["pm25"], bins=50, kde=True, color="darkred")
    plt.title("Distribution of PM2.5 Concentrations in Karachi", fontsize=16)
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "pm25_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Monthly PM2.5 Trend (Seasonality)
    logger.info("Generating Monthly PM2.5 Trend Plot...")
    monthly_trend = df.groupby(df["date"].dt.month)["pm25"].mean().reset_index()
    monthly_trend["date"] = monthly_trend["date"].apply(lambda x: pd.to_datetime(str(x), format="%m").strftime("%b"))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="date", y="pm25", data=monthly_trend, palette="YlOrBr")
    plt.title("Average PM2.5 by Month (Seasonality)", fontsize=16)
    plt.xlabel("Month")
    plt.ylabel("Average PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "monthly_pm25_trend.png"), dpi=300)
    plt.close()

    # 4. Scatter: Aerosol Optical Depth vs PM2.5
    logger.info("Generating AOD vs PM2.5 Scatter Plot...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Optical_Depth_047", y="pm25", data=df, alpha=0.3, color="teal")
    plt.title("MODIS Aerosol Optical Depth (0.47) vs Surface PM2.5", fontsize=16)
    plt.xlabel("MODIS AOD 0.47")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "aod_vs_pm25_scatter.png"), dpi=300)
    plt.close()

    logger.info("✅ All EDA visualizations generated successfully in reports/figures/")

if __name__ == "__main__":
    main()
