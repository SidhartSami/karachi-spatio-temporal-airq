import pandas as pd
import numpy as np

# Test loading the master dataset
df = pd.read_csv('data/processed/master_dataset.csv')
print(f'Master dataset loaded: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'PM2.5 column exists: {"pm25" in df.columns}')
print(f'PM2.5 range: {df["pm25"].min():.1f} - {df["pm25"].max():.1f} µg/m³')
print(f'Sample PM2.5 values: {df["pm25"].head(5).tolist()}')
