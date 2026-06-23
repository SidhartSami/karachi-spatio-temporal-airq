import argparse
import os
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

def load_gee_csv(pattern, label, value_cols, raw_dir='data/raw'):
    """Load a GEE-exported CSV from raw_dir/ and ensure value columns exist.

    `pattern` is a glob fragment matched against filenames. The _v2 GEE exports
    use the long band name (e.g. `absorbing_aerosol_index_v2.csv`) which does
    not contain the short label (`aer_ai`); accept the long name as an
    additional glob if it appears in any value_col.
    """
    extra_patterns = [v for v in value_cols if v and v != pattern]
    patterns = [pattern] + extra_patterns
    matches = []
    seen = set()
    for pat in patterns:
        for m in Path(raw_dir).glob(f'*{pat}*.csv'):
            if m.name not in seen:
                matches.append(m)
                seen.add(m.name)

    # Priority: _citywide > _v2 > original
    citywide_matches = [m for m in matches if '_citywide' in m.name]
    v2_matches = [m for m in matches if '_v2' in m.name]

    if citywide_matches:
        matches = citywide_matches
    elif v2_matches:
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
        
    if 'station' in df.columns:
        if df['station'].nunique() == 1 and str(df['station'].iloc[0]).lower() == 'karachi':
            df = df.drop(columns=['station'])
    
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
    if 'date' in df.columns:
        before_count = len(df)
        # Only aggregate if we actually have value columns to aggregate
        cols_to_agg = [c for c in value_cols if c in df.columns]
        if cols_to_agg:
            if 'station' in df.columns:
                df = df.groupby(['date', 'station'])[cols_to_agg].mean().reset_index()
            else:
                df = df.groupby(['date'])[cols_to_agg].mean().reset_index()
            after_count = len(df)
            if before_count != after_count:
                print(f'  ℹ️  Aggregated {before_count} -> {after_count} rows (multiple daily orbits/points)')

    print(f'  ✓ {label:<25}: {df.shape[0]:>6} rows, {df.shape[1]} cols → {matches[0].name}')
    return df

def main(raw_dir: str = 'data/raw', out_path: str = 'data/processed/merged_karachi_dataset.csv'):
    print(f'📂 Loading GEE exports from {raw_dir}/...')

    # S5P Value column names (as expected from asset band names)
    df_aer   = load_gee_csv('aer_ai',  'S5P Aerosol Index',     ['absorbing_aerosol_index'], raw_dir=raw_dir)
    df_no2   = load_gee_csv('no2',     'S5P NO2',               ['NO2_column_number_density'], raw_dir=raw_dir)
    df_so2   = load_gee_csv('so2',     'S5P SO2',               ['SO2_column_number_density'], raw_dir=raw_dir)
    df_co    = load_gee_csv('co',      'S5P CO',                ['CO_column_number_density'], raw_dir=raw_dir)
    
    # MODIS (Note: run_data_collection uses Optical_Depth_047/055)
    df_modis = load_gee_csv('modis',   'MODIS AOD',             ['Optical_Depth_047', 'Optical_Depth_055'], raw_dir=raw_dir)

    # ERA5 (Meteo)
    df_era5  = load_gee_csv('era5',    'ERA5 Meteorology',      ['wind_speed_10m', 'relative_humidity', 'temperature_2m'], raw_dir=raw_dir)
    if df_era5 is not None:
        df_era5 = df_era5.rename(columns={'wind_speed_10m': 'wind_speed', 'relative_humidity': 'rh'})

    # Optional
    df_viirs = load_gee_csv('viirs',   'VIIRS Nighttime Light', ['avg_rad', 'mean'], raw_dir=raw_dir)
    df_s2    = load_gee_csv('ndvi',    'Sentinel-2 NDVI/NDBI',  ['NDVI', 'NDBI'], raw_dir=raw_dir)

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

    def smart_merge(df1, df2, how='outer'):
        if df1 is None or df1.empty: return df2.copy() if df2 is not None else None
        if df2 is None or df2.empty: return df1.copy() if df1 is not None else None
        
        merge_keys = ['date']
        has_station1 = 'station' in df1.columns
        has_station2 = 'station' in df2.columns
        
        if has_station1 and has_station2:
            merge_keys.append('station')
            return pd.merge(df1, df2, on=merge_keys, how=how)
        elif has_station1 and not has_station2:
            return pd.merge(df1, df2, on='date', how='left')
        elif not has_station1 and has_station2:
            return pd.merge(df2, df1, on='date', how='left')
        else:
            return pd.merge(df1, df2, on='date', how=how)

    # Merge S5P data
    merged = pd.DataFrame()
    for next_df in station_dfs:
        merged = smart_merge(merged, next_df, how='outer')

    # ── Merge ERA5 (Broadcast to all stations) ────────────────────────────────
    if df_era5 is not None:
        era5_cols = ['date', 'wind_speed', 'rh', 'temperature_2m']
        if 'station' in df_era5.columns: era5_cols.append('station')
        era5_clean = df_era5[[c for c in era5_cols if c in df_era5.columns]]
        merged = smart_merge(merged, era5_clean, how='left')

    # ── Merge MODIS ───────────────────────────────────────────────────────────
    if df_modis is not None:
        modis_cols = ['date', 'Optical_Depth_047', 'Optical_Depth_055']
        if 'station' in df_modis.columns: modis_cols.append('station')
        modis_clean = df_modis[[c for c in modis_cols if c in df_modis.columns]]
        merged = smart_merge(merged, modis_clean, how='left')

    # ── Merge VIIRS (Monthly) ─────────────────────────────────────────────────
    # C5 fix (ISSUES_FOUND.md): prior version broadcast each month's NTL value
    # to every day of the *same* month. That means day 1 of March used March's
    # average NTL — a value not knowable until month-end. We now lag by 1
    # month so day D in month M uses month M-1's NTL (the value actually
    # available at time D).
    if df_viirs is not None:
        # Some GEE exports give us `year_month` (e.g. "2019-01") instead of a
        # real `date` column. Synthesise a date so the lag logic below works.
        if 'date' not in df_viirs.columns and 'year_month' in df_viirs.columns:
            df_viirs['date'] = pd.to_datetime(df_viirs['year_month'] + '-01')

        if 'date' in df_viirs.columns:
            df_viirs['year_month'] = df_viirs['date'].dt.strftime('%Y-%m')

        if 'year_month' in df_viirs.columns and not merged.empty:
            # Lag VIIRS by one month: a day in 2021-03 sees 2021-02's NTL.
            df_viirs['year_month_for_lag'] = (
                df_viirs['date'] - pd.DateOffset(months=1)
            ).dt.strftime('%Y-%m')

            merged['year_month_for_lag'] = merged['date'].dt.strftime('%Y-%m')

            ntl_col = 'avg_rad' if ('avg_rad' in df_viirs.columns and df_viirs['avg_rad'].notnull().any()) else 'mean'

            viirs_cols = ['year_month_for_lag']
            if 'station' in df_viirs.columns: viirs_cols.append('station')
            viirs_cols.append(ntl_col)

            viirs_subset = df_viirs[[c for c in viirs_cols if c in df_viirs.columns]]

            merge_on = ['year_month_for_lag']
            if 'station' in merged.columns and 'station' in viirs_subset.columns:
                merge_on.append('station')
            elif 'station' in viirs_subset.columns and 'station' not in merged.columns:
                # If merged is citywide but viirs has stations, average viirs to citywide
                viirs_subset = viirs_subset.groupby('year_month_for_lag')[ntl_col].mean().reset_index()

            merged = pd.merge(merged, viirs_subset.rename(columns={ntl_col: 'viirs_ntl'}),
                              on=merge_on, how='left')
            merged.drop(columns='year_month_for_lag', inplace=True)

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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'\n✅ SUCCESS: Dataset saved to {out_path}')
    print(f'   Final Shape: {df.shape}')

    if df[pollutant_cols].isnull().all().all():
        print("\n🛑 CRITICAL: The dataset is mostly NaNs. We need to re-run the GEE data collection.")
        print("   The S5P points were likely cloud-masked or scale was too small.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge GEE satellite + meteo exports into one daily-per-station CSV.")
    parser.add_argument("--raw-dir",  default="data/raw",
                        help="Directory containing GEE-exported CSVs (default: data/raw).")
    parser.add_argument("--out-path", default="data/processed/merged_karachi_dataset.csv",
                        help="Output CSV path.")
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, out_path=args.out_path)
