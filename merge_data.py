import pandas as pd
import numpy as np
from pathlib import Path
import os
import holidays
from datetime import datetime

def load_gee_csv(pattern, label, value_cols):
    """Load a GEE-exported CSV from data/raw/ and ensure value columns exist."""
    matches = list(Path('data/raw').glob(f'*{pattern}*.csv'))
    # Prioritize _v2 files
    v2_matches = [m for m in matches if '_v2' in m.name]
    if v2_matches:
        matches = v2_matches
    
    if not matches:
        print(f'  ⚠️  No file matching *{pattern}*.csv in data/raw/')
        return None
    
    try:
        df = pd.read_csv(matches[0])
    except Exception as e:
        print(f'  ❌ Error reading {matches[0].name}: {e}')
        return None

    # Parse date column
    date_col = [c for c in df.columns if 'date' in c.lower()]
    if date_col:
        df[date_col[0]] = pd.to_datetime(df[date_col[0]])
        df = df.rename(columns={date_col[0]: 'date'})
    
    # Normalize station/location column
    if 'location' in df.columns and 'station' not in df.columns:
        df = df.rename(columns={'location': 'station'})
    
    # Check for missing value columns
    available_cols = df.columns.tolist()
    missing_cols = [c for c in value_cols if c not in available_cols]
    
    if missing_cols:
        print(f'  ⚠️  Warning: Columns {missing_cols} not found in {matches[0].name}')
        # Check if 'mean' or other generic names exist
        if 'mean' in available_cols and len(value_cols) == 1:
            print(f'     Mapping "mean" -> "{value_cols[0]}"')
            df = df.rename(columns={'mean': value_cols[0]})
        else:
            # Add empty columns to avoid crashes
            for c in missing_cols:
                df[c] = np.nan
    
    # Check for actual data
    non_null_count = df[value_cols].notnull().sum().sum()
    if non_null_count == 0:
        print(f'  ⚠️  ALERT: {label} dataset contains 0 valid measurements (all pixels were likely masked).')
    
    # --- Aggregate by date/station to handle multiple orbits/granules ---
    if 'date' in df.columns and 'station' in df.columns:
        before_count = len(df)
        # Only aggregate if we actually have value columns to aggregate
        cols_to_agg = [c for c in value_cols if c in df.columns]
        if cols_to_agg:
            df = df.groupby(['date', 'station'])[cols_to_agg].mean().reset_index()
            after_count = len(df)
            if before_count != after_count:
                print(f'  ℹ️  Aggregated {before_count} -> {after_count} rows (multiple daily orbits/points)')

    print(f'  ✓ {label:<25}: {df.shape[0]:>6} rows, {df.shape[1]} cols → {matches[0].name}')
    return df

def main():
    print('📂 Loading GEE exports from data/raw/...')
    
    # S5P Value column names (as expected from asset band names)
    df_aer   = load_gee_csv('aer_ai',  'S5P Aerosol Index',     ['absorbing_aerosol_index'])
    df_no2   = load_gee_csv('no2',     'S5P NO2',               ['NO2_column_number_density'])
    df_so2   = load_gee_csv('so2',     'S5P SO2',               ['SO2_column_number_density'])
    df_co    = load_gee_csv('co',      'S5P CO',                ['CO_column_number_density'])
    
    # MODIS (Note: run_data_collection uses Optical_Depth_047/055)
    df_modis = load_gee_csv('modis',   'MODIS AOD',             ['Optical_Depth_047', 'Optical_Depth_055'])
    
    # ERA5 (Meteo)
    df_era5  = load_gee_csv('era5',    'ERA5 Meteorology',      ['wind_speed_10m', 'relative_humidity', 'temperature_2m'])
    if df_era5 is not None:
        df_era5 = df_era5.rename(columns={'wind_speed_10m': 'wind_speed', 'relative_humidity': 'rh'})
    
    # Optional
    df_viirs = load_gee_csv('viirs',   'VIIRS Nighttime Light', ['avg_rad', 'mean'])
    df_s2    = load_gee_csv('ndvi',    'Sentinel-2 NDVI/NDBI',  ['NDVI', 'NDBI'])

    print('\n🔗 Building master merged dataset...')
    
    # ── Merge S5P Daily Data ──────────────────────────────────────────────────
    station_dfs = []
    rename_map = {
        'absorbing_aerosol_index': 'aer_ai',
        'NO2_column_number_density': 'no2',
        'SO2_column_number_density': 'so2',
        'CO_column_number_density': 'co'
    }
    
    for df_key, col in [('df_aer', 'absorbing_aerosol_index'), 
                        ('df_no2', 'NO2_column_number_density'), 
                        ('df_so2', 'SO2_column_number_density'), 
                        ('df_co',  'CO_column_number_density')]:
        df = locals().get(df_key)
        if df is not None:
            # Rename column for brevity
            df = df.rename(columns={col: rename_map[col]})
            # Keep only necessary columns
            cols_to_keep = [c for c in ['date', 'station', rename_map[col]] if c in df.columns]
            station_dfs.append(df[cols_to_keep])

    if not station_dfs:
        print("❌ No S5P data found. Please check data/raw/ CSVs.")
        return

    from functools import reduce
    # Merge on date and station
    merged = station_dfs[0]
    for next_df in station_dfs[1:]:
        merged = pd.merge(merged, next_df, on=['date', 'station'], how='outer')

    # ── Merge ERA5 (Broadcast to all stations) ────────────────────────────────
    if df_era5 is not None:
        era5_cols = ['date', 'wind_speed', 'rh', 'temperature_2m']
        era5_clean = df_era5[[c for c in era5_cols if c in df_era5.columns]]
        merged = pd.merge(merged, era5_clean, on='date', how='left')

    # ── Merge MODIS ───────────────────────────────────────────────────────────
    if df_modis is not None:
        modis_cols = ['date', 'Optical_Depth_047', 'Optical_Depth_055']
        modis_clean = df_modis[[c for c in modis_cols if c in df_modis.columns]]
        merged = pd.merge(merged, modis_clean, on='date', how='left')

    # ── Merge VIIRS (Monthly) ─────────────────────────────────────────────────
    if df_viirs is not None:
        if 'date' in df_viirs.columns:
            df_viirs['year_month'] = df_viirs['date'].dt.strftime('%Y-%m')
        
        if 'year_month' in df_viirs.columns:
            merged['year_month'] = merged['date'].dt.strftime('%Y-%m')
            # Use 'avg_rad' or 'mean' as the NTL column
            ntl_col = 'avg_rad' if 'avg_rad' in df_viirs.columns else 'mean'
            viirs_subset = df_viirs[['year_month', 'station', ntl_col]]
            merged = pd.merge(merged, viirs_subset.rename(columns={ntl_col: 'viirs_ntl'}), 
                              on=['year_month', 'station'], how='left')
            merged.drop(columns='year_month', inplace=True)

    # ⚙️ Feature Engineering
    print('⚙️  Engineering temporal and Pakistan-specific features...')
    
    # Drop rows where we have absolutely no satellite data (optional, but keeps it clean)
    pollutant_cols = ['aer_ai', 'no2', 'so2', 'co']
    df = merged.dropna(subset=[c for c in pollutant_cols if c in merged.columns], how='all')
    
    if df.empty:
        print("⚠️ Warning: Merged dataset is empty because all satellite data is missing.")
        df = merged # Keep the empty/nan shell for now
    
    df = df.copy().sort_values(['station', 'date'])
    
    # 1. Cyclical time features
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    # 2. Pakistan Holidays
    pk_holidays = holidays.PK()
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in pk_holidays else 0)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Save
    os.makedirs('data/processed', exist_ok=True)
    out_file = 'data/processed/merged_karachi_dataset.csv'
    df.to_csv(out_file, index=False)
    print(f'\n✅ SUCCESS: Dataset saved to {out_file}')
    print(f'   Final Shape: {df.shape}')
    
    if df[pollutant_cols].isnull().all().all():
        print("\n🛑 CRITICAL: The dataset is mostly NaNs. We need to re-run the GEE data collection.")
        print("   The S5P points were likely cloud-masked or scale was too small.")

if __name__ == "__main__":
    main()
