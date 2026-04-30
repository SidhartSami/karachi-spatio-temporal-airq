"""
Create Master Dataset with PM2.5 Target
========================================
Creates master_dataset.csv with synthetic PM2.5 values based on realistic relationships
with available satellite and meteorological features.
"""

import pandas as pd
import numpy as np

def create_pm25_target(row):
    """Generate realistic PM2.5 values based on feature relationships"""
    base_pm25 = 45  # Base level for Karachi
    
    # AOD contribution (higher AOD = higher PM2.5)
    aod_contrib = 0
    if not pd.isna(row['Optical_Depth_055']):
        aod_contrib = row['Optical_Depth_055'] * 80
    
    # NO2 contribution
    no2_contrib = 0
    if not pd.isna(row['no2']):
        no2_contrib = row['no2'] * 15
    
    # SO2 contribution  
    so2_contrib = 0
    if not pd.isna(row['so2']):
        so2_contrib = row['so2'] * 10
    
    # Wind speed inverse (lower wind = higher PM2.5)
    wind_contrib = 0
    if not pd.isna(row['wind_speed']):
        wind_contrib = max(0, 15 - row['wind_speed']) * 2
    
    # Humidity contribution
    humidity_contrib = 0
    if not pd.isna(row['rh']):
        humidity_contrib = (row['rh'] - 50) * 0.3 if row['rh'] > 50 else 0
    
    # Seasonal effect (higher in winter, lower in monsoon)
    month = row['month']
    seasonal_contrib = 0
    if month in [11, 12, 1, 2]:  # Winter
        seasonal_contrib = 20
    elif month in [6, 7, 8]:  # Monsoon
        seasonal_contrib = -15
    
    # Add realistic noise
    noise = np.random.normal(0, 5)
    
    # Calculate final PM2.5 with bounds
    pm25 = base_pm25 + aod_contrib + no2_contrib + so2_contrib + wind_contrib + humidity_contrib + seasonal_contrib + noise
    return max(15, min(200, pm25))

def main():
    print("Creating master dataset with PM2.5 target...")
    
    # Load merged dataset
    df = pd.read_csv('data/processed/merged_karachi_dataset.csv')
    print(f"Loaded dataset: {df.shape}")
    
    # Create PM2.5 target
    df['pm25'] = df.apply(create_pm25_target, axis=1)
    
    # Statistics
    print(f"PM2.5 range: {df['pm25'].min():.1f} - {df['pm25'].max():.1f} µg/m³")
    print(f"Mean PM2.5: {df['pm25'].mean():.1f} µg/m³")
    
    # Save master dataset
    df.to_csv('data/processed/master_dataset.csv', index=False)
    print("Saved: data/processed/master_dataset.csv")
    
    # Show sample
    print("\nSample data:")
    print(df[['date', 'station', 'pm25', 'Optical_Depth_055', 'wind_speed', 'month']].head(3))

if __name__ == "__main__":
    main()
